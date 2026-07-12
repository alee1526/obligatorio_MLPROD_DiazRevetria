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
    loss: str = "weighted_ce"
    focal_gamma: float = 2.0
    lr_factor: float = 0.5
    lr_patience: int = 2
    min_lr: float = 1e-6
    patience: int = 4
    aug_rotation: float = 20
    aug_translate: float = 0.05
    aug_scale: float = 0.1
    aug_color: float = 0.1
    aug_erase: float = 0.0
    n_trials: int = 25

    @classmethod
    def load(cls, path=CONFIG_PATH):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def as_dict(self):
        return asdict(self)

    def aug_params(self):
        return {"rotation": self.aug_rotation, "translate": self.aug_translate,
                "scale": self.aug_scale, "color": self.aug_color, "erase": self.aug_erase}
