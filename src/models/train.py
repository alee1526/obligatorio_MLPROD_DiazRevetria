import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset_utils import CLASSES, DATA_DIR
from src.models.architectures import build_model
from src.models.config import Config
from src.models.dataset import CLASS_TO_IDX, MultimodalDataset
from src.models.metrics import compute_metrics

MODELS_DIR = DATA_DIR.parent / "models"


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


def main():
    cfg = Config.load()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    print("device:", device)

    train_ds = MultimodalDataset("train")
    val_ds = MultimodalDataset("val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    arch_kwargs = {
        "n_tabular_features": len(train_ds.feature_cols),
        "n_classes": len(CLASSES),
        "backbone": cfg.backbone,
        "unfreeze_blocks": cfg.unfreeze_blocks,
        "tab_hidden": cfg.tab_hidden,
        "tab_out": cfg.tab_out,
        "fusion_hidden": cfg.fusion_hidden,
        "dropout": cfg.dropout,
    }
    model = build_model(cfg.architecture, **arch_kwargs).to(device)

    labels = np.array([CLASS_TO_IDX[c] for c in train_ds.df["diagnostic"]])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(labels, len(CLASSES)).to(device))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    mlflow.set_experiment("skin-lesion")
    with mlflow.start_run():
        mlflow.log_params(cfg.as_dict())
        for epoch in range(1, cfg.epochs + 1):
            tr_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, device)
            m = compute_metrics(y_true, y_pred)
            mlflow.log_metrics({
                "train_loss": tr_loss, "val_loss": val_loss,
                "val_macro_f1": m["macro_f1"], "val_balanced_acc": m["balanced_acc"],
            }, step=epoch)
            print(f"epoch {epoch}: train_loss={tr_loss:.3f} val_loss={val_loss:.3f} "
                  f"macro_f1={m['macro_f1']:.3f} bal_acc={m['balanced_acc']:.3f}")
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                torch.save({
                    "state_dict": model.state_dict(),
                    "architecture": cfg.architecture,
                    "arch_kwargs": arch_kwargs,
                    "classes": list(CLASS_TO_IDX),
                    "feature_cols": train_ds.feature_cols,
                }, MODELS_DIR / "model.pt")
        mlflow.log_metric("best_val_macro_f1", best_f1)
    print(f"mejor macro-F1 en val: {best_f1:.3f} -> {MODELS_DIR / 'model.pt'}")


if __name__ == "__main__":
    main()
