import torch.nn as nn


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
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# implement forward method with token masking...