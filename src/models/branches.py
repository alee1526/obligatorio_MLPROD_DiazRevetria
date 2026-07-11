import timm
import torch.nn as nn


class ImageBranch(nn.Module):
    def __init__(self, backbone="efficientnet_b0", unfreeze_blocks=0):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.out_features = self.model.num_features
        self._set_trainable(unfreeze_blocks)

    def _set_trainable(self, unfreeze_blocks):
        for p in self.model.parameters():
            p.requires_grad = False
        if unfreeze_blocks < 0:
            for p in self.model.parameters():
                p.requires_grad = True
        elif unfreeze_blocks > 0:
            for p in self.model.blocks[-unfreeze_blocks:].parameters():
                p.requires_grad = True
            for name in ("conv_head", "bn2"):
                module = getattr(self.model, name, None)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = True

    def forward(self, x):
        return self.model(x)


class TabularBranch(nn.Module):
    def __init__(self, in_features, hidden=64, out_features=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
            nn.ReLU(),
        )
        self.out_features = out_features

    def forward(self, x):
        return self.net(x)
