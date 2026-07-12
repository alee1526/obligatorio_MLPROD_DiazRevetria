from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.dataset_utils import CLASSES, PROCESSED_DIR, TARGET
from src.features.image_utils import PROCESSED_IMAGES_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NON_FEATURE = {"img_id", "split", TARGET}


def build_transform(train, rotation=20, translate=0.05, scale=0.1, color=0.1, erase=0.0):
    steps = []
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(rotation),
            transforms.RandomAffine(0, translate=(translate, translate), scale=(1 - scale, 1 + scale)),
            transforms.ColorJitter(brightness=color, contrast=color, saturation=color),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train and erase > 0:
        steps.append(transforms.RandomErasing(p=erase))
    return transforms.Compose(steps)


class MultimodalDataset(Dataset):
    def __init__(self, split, tabular_path=PROCESSED_DIR / "tabular.parquet",
                 images_dir=PROCESSED_IMAGES_DIR, aug=None):
        df = pd.read_parquet(tabular_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.feature_cols = [c for c in df.columns if c not in NON_FEATURE]
        self.transform = build_transform(split == "train", **(aug or {}))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = self.transform(Image.open(self.images_dir / row["img_id"]).convert("RGB"))
        tabular = torch.tensor(row[self.feature_cols].to_numpy(dtype="float32"))
        label = CLASS_TO_IDX[row[TARGET]]
        return image, tabular, label


def main():
    for split in ["train", "val", "test"]:
        ds = MultimodalDataset(split)
        print(f"{split}: {len(ds)} muestras | {len(ds.feature_cols)} features tabulares")
    image, tabular, label = MultimodalDataset("train")[0]
    print(f"ejemplo -> imagen {tuple(image.shape)} | tabular {tuple(tabular.shape)} | label {label}")


if __name__ == "__main__":
    main()
