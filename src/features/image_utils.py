from PIL import Image

from src.data.dataset_utils import PROCESSED_DIR

IMG_SIZE = 224
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"


def resize_image(src, dst, size=IMG_SIZE):
    with Image.open(src) as im:
        im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(dst)
