import torch
import torch.nn as nn
from models.dinov2 import vit_base, vit_tiny, vit_small
from data.dataset import KDdataset, MIADDataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from models.student import StudentFKD, StudentFKDv2
from models.teacher import TeacherFKD
from utils.utils import WarmCosineScheduler
import numpy as np
import random
import torch.nn.functional as F
import time

# class_list = ["electrical_insulator", "metal_welding", "photovoltaic_module", "wind_turbine"]
class_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]

train_dataset = MIADDataset(dataset_path="./input/datasets/alfbom27/miad-ad", class_list=class_list, mode="train")


FROM_CHECKPOINT = True
CHECKPOINT_PATH = ""
NUM_ITERATIONS = 40000
WARMUP_ITERATIONS = 0.05*NUM_ITERATIONS
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8,
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

student = StudentFKDv2(backbone=student_backbone, teacher_dims=384, student_dims=192)
# student = Student(backbone=student_backbone)
student = student.to(DEVICE)

lr = 1e-3
# optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-3)
optimizer = AdamW([
    {"params": student.backbone.parameters(), "lr": lr, "weight_decay": 1e-4},
    {"params": student.adapters.parameters(), "lr": lr, "weight_decay": 1e-4}
])

lr_scheduler = WarmCosineScheduler(optimizer, base_value=lr, final_value=1e-4, total_iters=NUM_ITERATIONS,
                                   warmup_iters=WARMUP_ITERATIONS)

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
while it < NUM_ITERATIONS:
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
            student_out, _ = student(images)

            loss = 0
            for sf, tf in zip(student_out, teacher_out):
                sf = F.normalize(sf, dim=1)
                tf = F.normalize(tf, dim=1)
                # loss += F.mse_loss(sf,tf)
                loss += (1 - (sf * tf).sum(dim=1)).mean()

            loss /= len(student_out)

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
            for sf, tf in zip(student_out, teacher_out):
                sf_norm = F.normalize(sf, dim=1)
                tf_norm = F.normalize(tf, dim=1)
                cos += (sf_norm * tf_norm).sum(dim=1).mean()

            cos /= len(student_out)

        cos_embedding.append(cos.item())

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
    }, "checkpoint_vit_tiny.pth")


