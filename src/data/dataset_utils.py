from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_PATH = RAW_DIR / "metadata.csv"
SPLITS_PATH = DATA_DIR / "splits.csv"
IMAGE_GLOB = "imgs_part_*/*.png"

TARGET = "diagnostic"
CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
MALIGNANT = {"BCC", "MEL", "SCC"}


def index_images(raw_dir=RAW_DIR):
    return {p.name: p for p in Path(raw_dir).glob(IMAGE_GLOB)}


def integrity_report(df, images):
    referenced = set(df["img_id"])
    missing = df.loc[df["img_path"].isna(), "img_id"].tolist()
    orphan = [name for name in images if name not in referenced]
    return {
        "n_rows": len(df),
        "n_images_found": int(df["img_path"].notna().sum()),
        "n_missing_images": len(missing),
        "n_orphan_images": len(orphan),
        "missing_examples": missing[:10],
        "orphan_examples": orphan[:10],
    }
