import torch.nn as nn
import torch
import math


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
        return en


class StudentFKD(nn.Module):
    def __init__(self, backbone, teacher_dims, student_dims, target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
                 fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]]):
        super().__init__()
        self.backbone = backbone
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder

        self.adapters = nn.ModuleList([
            nn.Conv2d(student_dims, teacher_dims, kernel_size=1, bias=False)
            for _ in range(len(fuse_layer_encoder))
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
        en_raw = en
        side = int(math.sqrt(en[0].shape[1] - 1 - self.backbone.num_register_tokens))

        en = [self.fuse_feature([en[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        en = [e[:, self.backbone.num_register_tokens + 1:, :] for e in en]
        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]

        projected = []
        for e, l in zip(en, self.adapters):
            projected.append(l(e))

        return projected, en_raw

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


