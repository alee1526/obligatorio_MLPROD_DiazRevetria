from dataclasses import asdict, dataclass, fields

import yaml

from src.data.dataset_utils import DATA_DIR

CONFIG_PATH = DATA_DIR.parent / "config.yaml"


@dataclass
class Config:
    architecture: str = "multimodal"
    backbone: str = "efficientnet_b0"
    unfreeze_blocks: int = 0
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    dropout: float = 0.3
    tab_hidden: int = 64
    tab_out: int = 32
    fusion_hidden: int = 128

    @classmethod
    def load(cls, path=CONFIG_PATH):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def as_dict(self):
        return asdict(self)
