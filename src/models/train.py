import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset_utils import CLASSES, DATA_DIR
from src.models.architectures import build_model
from src.models.config import Config
from src.models.dataset import CLASS_TO_IDX, MultimodalDataset
from src.models.losses import build_loss
from src.models.metrics import compute_metrics

MODELS_DIR = DATA_DIR.parent / "models"
MLFLOW_URI = str(DATA_DIR.parent / "mlruns")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes)
    weights = len(labels) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for image, tabular, label in loader:
            image, tabular, label = image.to(device), tabular.to(device), label.to(device)
            logits = model(image, tabular)
            loss = criterion(logits, label)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * label.size(0)
            preds.append(logits.argmax(1).cpu())
            targets.append(label.cpu())
    return total_loss / len(loader.dataset), torch.cat(targets).numpy(), torch.cat(preds).numpy()


def arch_kwargs_from(cfg, n_tabular_features):
    return {
        "n_tabular_features": n_tabular_features,
        "n_classes": len(CLASSES),
        "backbone": cfg.backbone,
        "unfreeze_blocks": cfg.unfreeze_blocks,
        "tab_hidden": cfg.tab_hidden,
        "tab_out": cfg.tab_out,
        "fusion_hidden": cfg.fusion_hidden,
        "dropout": cfg.dropout,
    }


def train_run(cfg, device, save_path=None, log=True, on_epoch=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train_ds = MultimodalDataset("train", aug=cfg.aug_params())
    val_ds = MultimodalDataset("val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    arch_kwargs = arch_kwargs_from(cfg, len(train_ds.feature_cols))
    model = build_model(cfg.architecture, **arch_kwargs).to(device)

    labels = np.array([CLASS_TO_IDX[c] for c in train_ds.df["diagnostic"]])
    weight = class_weights(labels, len(CLASSES)).to(device)
    criterion = build_loss(cfg.loss, weight=weight, gamma=cfg.focal_gamma)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)

    best_f1, bad = -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, device)
        m = compute_metrics(y_true, y_pred)
        scheduler.step(m["macro_f1"])
        if log:
            mlflow.log_metrics({
                "train_loss": tr_loss, "val_loss": val_loss,
                "val_macro_f1": m["macro_f1"], "val_balanced_acc": m["balanced_acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
        print(f"epoch {epoch}: train_loss={tr_loss:.3f} val_loss={val_loss:.3f} "
              f"macro_f1={m['macro_f1']:.3f} bal_acc={m['balanced_acc']:.3f}")
        if on_epoch is not None:
            on_epoch(epoch, m)
        if m["macro_f1"] > best_f1:
            best_f1, bad = m["macro_f1"], 0
            if save_path is not None:
                torch.save({
                    "state_dict": model.state_dict(),
                    "architecture": cfg.architecture,
                    "arch_kwargs": arch_kwargs,
                    "classes": list(CLASS_TO_IDX),
                    "feature_cols": train_ds.feature_cols,
                }, save_path)
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                print(f"early stopping en época {epoch}")
                break
    return best_f1


def main():
    cfg = Config.load()
    device = get_device()
    print("device:", device)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("skin-lesion")
    with mlflow.start_run():
        mlflow.log_params(cfg.as_dict())
        best_f1 = train_run(cfg, device, save_path=MODELS_DIR / "model.pt")
        mlflow.log_metric("best_val_macro_f1", best_f1)
        mlflow.log_artifact(str(MODELS_DIR / "model.pt"))
    print(f"mejor macro-F1 en val: {best_f1:.3f} -> {MODELS_DIR / 'model.pt'}")


if __name__ == "__main__":
    main()
