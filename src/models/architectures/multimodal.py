import torch
import torch.nn as nn

from src.models.branches import ImageBranch, TabularBranch


class MultimodalModel(nn.Module):
    def __init__(self, n_tabular_features, n_classes=6, backbone="efficientnet_b0",
                 unfreeze_blocks=0, tab_hidden=64, tab_out=32, fusion_hidden=128, dropout=0.3):
        super().__init__()
        self.image = ImageBranch(backbone, unfreeze_blocks)
        self.tabular = TabularBranch(n_tabular_features, tab_hidden, tab_out)
        fused = self.image.out_features + self.tabular.out_features
        self.head = nn.Sequential(
            nn.Linear(fused, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, n_classes),
        )

    def forward(self, image, tabular):
        img_feat = self.image(image)
        tab_feat = self.tabular(tabular)
        return self.head(torch.cat([img_feat, tab_feat], dim=1))
