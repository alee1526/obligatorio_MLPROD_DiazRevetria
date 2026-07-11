import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data.dataset import load_with_splits
from src.data.dataset_utils import DATA_DIR
from src.features.image_utils import IMG_SIZE
from src.features.tabular_utils import FEATURES, select_features
from src.models.architectures import build_model
from src.models.dataset import IMAGENET_MEAN, IMAGENET_STD

MODELS_DIR = DATA_DIR.parent / "models"

SERVE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_bundle(model_path=MODELS_DIR / "model.pt",
                transformer_path=MODELS_DIR / "tabular_transformer.joblib"):
    ckpt = torch.load(model_path, map_location="cpu")
    model = build_model(ckpt["architecture"], **ckpt["arch_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    transformer = joblib.load(transformer_path)
    return model, transformer, ckpt["classes"]


def predict(image_path, clinical, model, transformer, classes):
    image = SERVE_TRANSFORM(Image.open(image_path).convert("RGB")).unsqueeze(0)
    X = select_features(pd.DataFrame([clinical]))
    tabular = torch.tensor(transformer.transform(X), dtype=torch.float32)
    with torch.no_grad():
        probs = F.softmax(model(image, tabular), dim=1).squeeze(0)
    return {c: float(p) for c, p in zip(classes, probs)}


def main():
    model, transformer, classes = load_bundle()
    row = load_with_splits()
    row = row[row["split"] == "test"].iloc[0]
    clinical = {c: row[c] for c in FEATURES}
    probs = predict(row["img_path"], clinical, model, transformer, classes)
    print(f"imagen: {row['img_id']} | real: {row['diagnostic']}")
    for c, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {c}: {p:.3f}")


if __name__ == "__main__":
    main()
