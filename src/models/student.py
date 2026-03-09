import torch.nn as nn
import torch

class StudentHead(nn.Module):
    def __init__(self, in_dim=192, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Student(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)

            if i in self.target_layers:
                en.append(x)

            if i == self.target_layers[-1]:
                break
        return en


class StudentFKD(nn.Module):
    def __init__(self, backbone, teacher_dims, student_dims, target_layers=[2, 3, 4, 5, 6, 7, 8, 9]):
        super().__init__()
        self.backbone = backbone
        self.target_layers = target_layers

        self.adapters = nn.ModuleList([
            nn.Linear(student_dims, teacher_dims, bias=False)
            for _ in target_layers
        ])

    def forward(self, x):
        x = self.backbone.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)

            if i in self.target_layers:
                en.append(x)

            if i == self.target_layers[-1]:
                break

        # en = [e[:, self.backbone.num_register_tokens + 1:, :] for e in en]

        projected = []
        for e, l in zip(en, self.adapters):
            projected.append(l(e))
        return projected


