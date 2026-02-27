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
import yaml
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to YAML config file",
)

args = parser.parse_args()

with open(args.config, 'r') as file:
    config=yaml.safe_load(file)

FROM_CHECKPOINT = config["from_checkpoint"]
CHECKPOINT_PATH = config["checkpoint_path"]
EPOCHS = config["epochs"]
WARMUP_EPOCHS = config["warmup_epochs"]
LEARNING_RATE = config["learning_rate"]
BATCH_SIZE = config["batch_size"]
DATASET_PATH = config["dataset_path"]
TEACHER_CHECKPOINT_PATH = config["teacher_checkpoint_path"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_list = ["electrical_insulator", "metal_welding", "photovoltaic_module", "wind_turbine"]

train_dataset = MIADDataset(dataset_path=DATASET_PATH, class_list=class_list, mode="train")


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

ckpt = torch.load(TEACHER_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
teacher_backbone.load_state_dict(ckpt, strict=True)

teacher = TeacherFKD(backbone=teacher_backbone)
teacher = teacher.to(DEVICE)

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
student = student.to(DEVICE)


optimizer = AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
lr_scheduler = WarmCosineScheduler(optimizer, base_value=LEARNING_RATE, final_value=1e-6, total_iters=EPOCHS * len(train_data),
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

print("Starting training...")
scaler = torch.cuda.amp.GradScaler()
while it < (EPOCHS*len(train_data)):
    train_loss = []
    cos_embedding = []

    student.train()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in train_data:
        images, _, _ = batch
        images = images.to(DEVICE)

        with torch.cuda.amp.autocast():
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
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        lr_scheduler.step()

        train_loss.append(loss.item())

        with torch.no_grad():
            cos = 0
            for sf, tf in zip(student_out, teacher_out):
                sf = F.normalize(sf, dim=-1)
                tf = F.normalize(tf, dim=-1)
                cos += F.cosine_similarity(sf, tf, dim=-1).mean().item()
            cos_embedding.append(cos/len(student_out))

        it += 1
        if it % 200 == 0:
            torch.cuda.synchronize()
            end = time.perf_counter()
            avg_it_time = (end - start) / 200
            start = time.perf_counter()
            print(
                f"iter [{it}/{EPOCHS * len(train_data)}], loss:{np.mean(train_loss):.8f}, lr: {optimizer.param_groups[0]['lr']:.8f}, avg cos embedding: {np.mean(cos_embedding):.5f}, avg it time: {avg_it_time:.4f}")
            train_loss = []
            cos_embedding = []
            it_times = []
            torch.save({
                "iteration": it,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            }, "checkpoint.pth")


    print(f"iter [{it}/{EPOCHS*len(train_data)}], loss:{np.mean(train_loss):.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}")
    torch.save({
        "iteration": it,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    }, "checkpoint.pth")


