import pandas as pd

from src.data.dataset_utils import (
    METADATA_PATH,
    RAW_DIR,
    SPLITS_PATH,
    index_images,
    integrity_report,
)


def load_dataset(metadata_path=METADATA_PATH, raw_dir=RAW_DIR):
    df = pd.read_csv(metadata_path)
    images = index_images(raw_dir)
    df["img_path"] = df["img_id"].map(lambda name: str(images.get(name)) if name in images else None)
    return df


def load_with_splits(splits_path=SPLITS_PATH):
    df = load_dataset()
    splits = pd.read_csv(splits_path)[["img_id", "split"]]
    return df.merge(splits, on="img_id", how="inner")


def main():
    df = load_dataset()
    report = integrity_report(df, index_images())
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
