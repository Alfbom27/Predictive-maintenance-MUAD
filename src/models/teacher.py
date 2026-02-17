import torch
import torch.nn as nn


class Teacher(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x
