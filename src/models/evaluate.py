import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.models.architectures import build_model
from src.models.dataset import MultimodalDataset
from src.models.metrics import compute_metrics
from src.models.train import MODELS_DIR, get_device


def load_checkpoint(model_path, device):
    ckpt = torch.load(model_path, map_location="cpu")
    model = build_model(ckpt["architecture"], **ckpt["arch_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def evaluate(model, loader, device):
    preds, targets = [], []
    with torch.no_grad():
        for image, tabular, label in loader:
            logits = model(image.to(device), tabular.to(device))
            preds.append(logits.argmax(1).cpu())
            targets.append(label)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODELS_DIR / "model.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.model, device)
    ds = MultimodalDataset(args.split)

    # El checkpoint fija el orden de las columnas tabulares con el que se entreno:
    # si el parquet cambio, las features entran corridas y las metricas no significan nada.
    if ckpt["feature_cols"] != ds.feature_cols:
        raise SystemExit(
            f"features del checkpoint != features de {args.split}: "
            f"{len(ckpt['feature_cols'])} vs {len(ds.feature_cols)} columnas"
        )

    y_true, y_pred = evaluate(model, DataLoader(ds, batch_size=args.batch_size), device)
    m = compute_metrics(y_true, y_pred)
    classes = ckpt["classes"]

    print(f"modelo: {args.model}")
    print(f"split: {args.split} | {len(ds)} muestras | device: {device}\n")
    print(f"macro_f1={m['macro_f1']:.3f} balanced_acc={m['balanced_acc']:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0))
    print("matriz de confusion (filas=real, columnas=predicho)")
    print("        " + "".join(f"{c:>6}" for c in classes))
    for c, row in zip(classes, confusion_matrix(y_true, y_pred, labels=range(len(classes)))):
        print(f"{c:>8}" + "".join(f"{v:>6}" for v in row))


if __name__ == "__main__":
    main()
