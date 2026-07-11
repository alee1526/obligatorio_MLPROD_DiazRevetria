import joblib
import pandas as pd

from src.data.dataset import load_dataset, load_with_splits
from src.data.dataset_utils import DATA_DIR, PROCESSED_DIR, TARGET
from src.features.image_utils import IMG_SIZE, PROCESSED_IMAGES_DIR, resize_image
from src.features.tabular import build_tabular_transformer
from src.features.tabular_utils import select_features

MODELS_DIR = DATA_DIR.parent / "models"


def preprocess_tabular():
    df = load_with_splits()
    X = select_features(df)
    train = df["split"] == "train"

    transformer = build_tabular_transformer()
    transformer.fit(X[train])

    features = pd.DataFrame(
        transformer.transform(X),
        columns=transformer.get_feature_names_out(),
        index=df.index,
    )
    features.insert(0, "img_id", df["img_id"].values)
    features["split"] = df["split"].values
    features[TARGET] = df[TARGET].values

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_DIR / "tabular.parquet", index=False)
    joblib.dump(transformer, MODELS_DIR / "tabular_transformer.joblib")

    print(f"tabular procesado: {features.shape} -> {PROCESSED_DIR / 'tabular.parquet'}")
    print(f"transformer ajustado -> {MODELS_DIR / 'tabular_transformer.joblib'}")


def preprocess_images(size=IMG_SIZE):
    df = load_dataset()
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        resize_image(row.img_path, PROCESSED_IMAGES_DIR / row.img_id, size)
        if i % 500 == 0 or i == n:
            print(f"  imágenes: {i}/{n}")
    print(f"imágenes RGB {size}x{size} -> {PROCESSED_IMAGES_DIR}")


def main():
    preprocess_tabular()
    preprocess_images()


if __name__ == "__main__":
    main()
