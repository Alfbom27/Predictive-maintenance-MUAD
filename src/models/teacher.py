import torch
import torch.nn as nn
import torch.nn.functional as F


class Teacher(nn.Module):
    def __init__(self, backbone, teacher_dims, student_dims, target_layers=[2, 3, 4, 5, 6, 7, 8, 9]):
        super().__init__()
        self.backbone = backbone

        self.target_layers = target_layers

        self.adapters = nn.ModuleList([
            nn.Linear(teacher_dims, student_dims, bias=False)
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
            projected.append(F.relu(l(e)))
        return projected

class TeacherFKD(nn.Module):
    def __init__(self, backbone, target_layers=[2, 3, 4, 5, 6, 7, 8, 9]):
        super().__init__()
        self.backbone = backbone
        self.target_layers = target_layers

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
        return en
