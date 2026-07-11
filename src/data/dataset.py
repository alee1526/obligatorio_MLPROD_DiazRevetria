import pandas as pd

from src.data.dataset_utils import (
    DATA_DIR,
    METADATA_PATH,
    index_images,
    integrity_report,
)


def load_dataset(metadata_path=METADATA_PATH, data_dir=DATA_DIR):
    df = pd.read_csv(metadata_path)
    images = index_images(data_dir)
    df["img_path"] = df["img_id"].map(lambda name: str(images.get(name)) if name in images else None)
    return df


def main():
    df = load_dataset()
    report = integrity_report(df, index_images())
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
