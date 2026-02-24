import torch
import torch.nn as nn
from models.dinov2 import vit_base, vit_tiny
from data.dataset import KDdataset, MIADDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.student import StudentFKD
from models.teacher import TeacherFKD
from utils.utils import WarmCosineScheduler
import numpy as np
import random
import torch.nn.functional as F


class_list = ["electrical_insulator", "metal_welding", "photovoltaic_module", "wind_turbine"]

train_dataset = MIADDataset(dataset_path="miad", class_list=class_list, mode="train")

FROM_CHECKPOINT = False
CHECKPOINT_PATH = ""
EPOCHS = 30
WARMUP_EPOCHS = 2
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
#                        pin_memory=True)

train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

teacher_backbone = vit_base(
    patch_size=14,
    img_size=518,
    block_chunks=0,
    init_values=1e-8,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
)

ckpt = torch.load("../PycharmProjects/Pythonprojects/Predictive-maintenance-MUAD/src/weights/dinov2_vitb14_pretrain.pth", map_location="cpu", weights_only=False)
teacher_backbone.load_state_dict(ckpt, strict=True)

teacher = TeacherFKD(backbone=teacher_backbone)

for p in teacher.parameters():
    p.requires_grad = False

teacher.eval()

student_backbone = vit_tiny(
    patch_size=14,
    img_size=518,
    block_chunks=0,
    init_values=1e-8,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
)

student = StudentFKD(backbone=student_backbone, teacher_dims=768, student_dims=192)



lr = 2e-3
optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-3)
lr_scheduler = WarmCosineScheduler(optimizer, base_value=lr, final_value=1e-6, total_iters=EPOCHS * len(train_data),
                                   warmup_iters=WARMUP_EPOCHS * len(train_data))

if FROM_CHECKPOINT:
    print("Continuing from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    student.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    it = checkpoint["iteration"]

else:
    it = 0

while it < (EPOCHS*len(train_data)):
    print("Starting training...")
    train_loss = []
    cos_embedding = []

    student.train()
    for batch in train_data:
        images, _, _ = batch
        images = images.to(DEVICE)

        # Teacher forward pass
        with torch.no_grad():
            teacher_out = teacher(images)

        # Student forward pass
        student_out = student(images)

        # loss
        loss = 0
        for sf, tf in zip(student_out, teacher_out):
            sf = F.normalize(sf, dim=-1)
            tf = F.normalize(tf, dim=-1)
            loss += F.mse_loss(sf, tf)
        loss /= len(student_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        it += 1
        lr_scheduler.step()

        train_loss.append(loss.item())

        with torch.no_grad():
            cos = 0
            for sf, tf in zip(student_out, teacher_out):
                sf = F.normalize(sf, dim=-1)
                tf = F.normalize(tf, dim=-1)
                cos += F.cosine_similarity(sf, tf, dim=-1).mean().item()
            cos_embedding.append(cos/len(student_out))

        if it % 100 == 0:
            print(
                f"iter [{it}/{EPOCHS * len(train_data)}], loss:{np.mean(train_loss):.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}, avg cos embedding: {np.mean(cos_embedding)}")


    print(f"iter [{it}/{EPOCHS*len(train_data)}], loss:{np.mean(train_loss):.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}")
    torch.save({
        "iteration": it,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    }, "checkpoint.pth")


