import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(name, weight=None, gamma=2.0):
    if name == "focal":
        return FocalLoss(weight=weight, gamma=gamma)
    return nn.CrossEntropyLoss(weight=weight)
