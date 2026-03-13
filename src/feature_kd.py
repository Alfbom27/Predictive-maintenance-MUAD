import torch
import torch.nn as nn
from models.dinov2 import vit_base, vit_tiny, vit_small
from data.dataset import KDdataset, MIADDataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from models.student import StudentFKD, Student
from models.teacher import TeacherFKD, Teacher
from utils.utils import WarmCosineScheduler
import numpy as np
import random
import torch.nn.functional as F
import time

class_list = ["electrical_insulator", "metal_welding", "photovoltaic_module", "wind_turbine"]

train_dataset = MIADDataset(dataset_path="./input/datasets/alfbom27/miad-ad", class_list=class_list, mode="train")

dataset_size = len(train_dataset)
val_size = int(0.1 * dataset_size)
train_size = dataset_size - val_size

train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
)

FROM_CHECKPOINT = True
CHECKPOINT_PATH = "./input/models/alfbom27/vit-tiny-11k-fm/pytorch/default/1/checkpoint_vit_tiny_11k.pth"
EPOCHS = 30
WARMUP_EPOCHS = 2
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_data = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                        pin_memory=True)

val_data = DataLoader(dataset=val_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                      pin_memory=True)
# train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

teacher_backbone = vit_small(
    patch_size=14,
    img_size=518,
    block_chunks=0,
    init_values=1e-8,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
)

ckpt = torch.load("./input/models/alfbom27/dinov2-small/pytorch/default/1/dinov2_vits14_pretrain.pth",
                  map_location="cpu", weights_only=False)
teacher_backbone.load_state_dict(ckpt, strict=True)

teacher = TeacherFKD(backbone=teacher_backbone)
# teacher = Teacher(backbone=teacher_backbone, teacher_dims=384, student_dims=192)
teacher = teacher.to(DEVICE)

for p in teacher.backbone.parameters():
    p.requires_grad = False

# teacher.eval()
teacher.backbone.eval()

student_backbone = vit_tiny(
    patch_size=14,
    img_size=518,
    block_chunks=0,
    init_values=1e-8,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
)

student = StudentFKD(backbone=student_backbone, teacher_dims=384, student_dims=192)
# student = Student(backbone=student_backbone)
student = student.to(DEVICE)

lr = 1e-3
# optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-3)
optimizer = AdamW([
    {"params": student.backbone.parameters(), "lr": lr, "weight_decay": 1e-4},
    {"params": student.adapters.parameters(), "lr": lr, "weight_decay": 1e-3}
])
# optimizer = AdamW(
#     list(student.parameters()) + list(teacher.adapters.parameters()),
#     lr=lr,
#     weight_decay=1e-3
# )
lr_scheduler = WarmCosineScheduler(optimizer, base_value=lr, final_value=1e-6, total_iters=EPOCHS * len(train_data),
                                   warmup_iters=WARMUP_EPOCHS * len(train_data))

if FROM_CHECKPOINT:
    print("Continuing from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    student.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    it = checkpoint["iteration"]

else:
    it = 0

scaler = torch.amp.GradScaler("cuda")
while it < (EPOCHS * len(train_data)):
    print("Starting training...")
    train_loss = []
    cos_embedding = []

    student.train()
    for batch in train_data:
        images, _, _ = batch
        images = images.to(DEVICE)

        with torch.amp.autocast("cuda"):
            # Teacher forward pass
            with torch.no_grad():
                teacher_out = teacher(images)
            # Student forward pass
            student_projected, student_raw = student(images)

            feat_loss = 0
            attn_loss = 0
            for sp, sf, tf in zip(student_projected, student_raw, teacher_out):
                # feat loss: projected student vs teacher (both 384-dim)
                sp_norm = F.normalize(sp, dim=1)
                tf_norm = F.normalize(tf, dim=1)
                feat_loss += (1 - (sp_norm * tf_norm).sum(dim=1)).mean()

                # attn loss: raw student (192) vs teacher (384)
                attn_s = (sf ** 2).sum(dim=1).flatten(1)  # (B, H*W)
                attn_t = (tf ** 2).sum(dim=1).flatten(1)  # (B, H*W)

                attn_s = attn_s / (attn_s.norm(dim=-1, keepdim=True) + 1e-8)
                attn_t = attn_t / (attn_t.norm(dim=-1, keepdim=True) + 1e-8)
                attn_loss += (1 - F.cosine_similarity(attn_s, attn_t, dim=-1)).mean()

            feat_loss /= len(student_projected)
            attn_loss /= len(student_raw)
            loss = feat_loss + 0.5 * attn_loss

        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        it += 1
        lr_scheduler.step()

        train_loss.append(loss.item())

        with torch.no_grad():
            cos = 0
            for i, (sf, tf) in enumerate(zip(student_raw, teacher_out)):
                attn_s = (sf ** 2).sum(dim=1).flatten(1)
                attn_t = (tf ** 2).sum(dim=1).flatten(1)

                s_norm = attn_s.norm(dim=-1)
                t_norm = attn_t.norm(dim=-1)

                attn_s = attn_s / (s_norm.unsqueeze(-1) + 1e-8)
                attn_t = attn_t / (t_norm.unsqueeze(-1) + 1e-8)

                cos_layer = F.cosine_similarity(attn_s, attn_t, dim=-1).mean()

        cos += cos_layer

        cos /= len(student_raw)

        if it % 100 == 0:
            print(
                f"iter [{it}/{EPOCHS * len(train_data)}], loss:{np.mean(train_loss):.8f}, lr: {optimizer.param_groups[0]['lr']:.10f}, avg cos: {np.mean(cos_embedding)}")

    print(
        f"iter [{it}/{EPOCHS * len(train_data)}], loss:{np.mean(train_loss):.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}")
    torch.save({
        "iteration": it,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    }, "./working/checkpoint.pth")


