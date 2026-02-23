import torch
import torch.nn as nn
from models.dinov1 import vit_base, vit_tiny, DINOHead
from data.dataset import KDdataset, MIADDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.student import Student, StudentHead
from models.teacher import Teacher
from utils.utils import WarmCosineScheduler
import numpy as np
import random
import torch.nn.functional as F
from utils.loss import RoBLoss

# Data

def multi_collate(batch):
    num_views = len(batch[0])
    out = []

    for view in range(num_views):
        out.append(torch.stack([sample[view] for sample in batch]))

    return out


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cosine_loss(s,t):
    return 2 - 2 * (s * t).sum(dim=-1).mean()

def get_param_groups(model):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def cosine_wd_scheduler(base_value, final_value, epochs, niter_per_epoch):
    total_iters = epochs * niter_per_epoch
    iters = np.arange(total_iters)

    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / total_iters)
    )

    return schedule

class_list = ["electrical_insulator", "metal_welding", "photovoltaic_module", "wind_turbine"]

miad_data = MIADDataset(dataset_path="miad", class_list=class_list, mode="train", kd_training=True)
train_dataset = KDdataset(miad_data)

FROM_CHECKPOINT = False
CHECKPOINT_PATH = ""
EPOCHS = 30
WARMUP_EPOCHS = 1
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
#                        pin_memory=True, collate_fn=multi_collate, persistent_workers=True, worker_init_fn=seed_worker)

train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=multi_collate)


# Student
student_backbone = vit_tiny(patch_size=16, drop_path_rate=0.1)
student_head = DINOHead(in_dim=192, out_dim=65536)
student = Student(backbone=student_backbone, head=student_head)
student = student.to(DEVICE)


ckpt = torch.load(
    "../PycharmProjects/Pythonprojects/Predictive-maintenance-MUAD/src/weights/dino_vitbase16_pretrain_full_checkpoint.pth",
    map_location="cpu")

backbone_dict = {
    k[len("backbone."):]: v
    for k, v in ckpt["teacher"].items()
    if k.startswith("backbone.")
}

head_dict = {
    k[len("head."):]: v
    for k, v in ckpt["teacher"].items()
    if k.startswith("head.")
}


teacher_backbone = vit_base(patch_size=16)
teacher_backbone.load_state_dict(backbone_dict)

teacher_head = DINOHead(
    in_dim=768,
    out_dim=65536,
)
teacher_head.load_state_dict(head_dict)

teacher = Teacher(backbone=teacher_backbone, head=teacher_head)
teacher = teacher.to(DEVICE)

for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()

# Optim and Lr scheduler

param_group = get_param_groups(student)

lr = 2e-3 / (1024/BATCH_SIZE)
# lr = 2e-3
optimizer = AdamW(param_group, lr=lr, weight_decay=4e-2)
lr_scheduler = WarmCosineScheduler(optimizer, base_value=lr, final_value=1e-6, total_iters=EPOCHS * len(train_data),
                                   warmup_iters=WARMUP_EPOCHS * len(train_data))
wd_scheduler = cosine_wd_scheduler(base_value=4e-2, final_value=0.4, epochs=EPOCHS, niter_per_epoch=len(train_data))
loss_fn = RoBLoss()

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
        global_crops = batch[:2]
        local_crops = batch[2:]
        B = global_crops[0].shape[0]

        # teacher pred
        with torch.no_grad():
            teacher_global = teacher(torch.cat(global_crops, dim=0).to(DEVICE))
            teacher_global = F.normalize(teacher_global, dim=-1)


        t1, t2 = teacher_global[:B], teacher_global[B:]

        student_global = student(torch.cat(global_crops, dim=0).to(DEVICE))
        student_global = F.normalize(student_global, dim=-1)

        student_local = student(torch.cat(local_crops, dim=0).to(DEVICE))
        student_local = F.normalize(student_local, dim=-1)

        s1, s2 = student_global[:B], student_global[B:]
        s_locals = student_local.chunk(len(local_crops)) # List of local crops over batch


        # Loss
        num_views = 2 + 2 * len(s_locals)
        loss = 0

        loss += loss_fn(s1, t1)
        loss += loss_fn(s2, t2)

        for s_v in s_locals:
            loss += loss_fn(s_v, t1) + loss_fn(s_v, t2)

        loss /= num_views


        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        it += 1
        lr_scheduler.step()
        optimizer.param_groups[0]["weight_decay"] = wd_scheduler[it]

        train_loss.append(loss.item())
        with torch.no_grad():
            cos1 = F.cosine_similarity(s1, t1, dim=-1).mean()
            cos2 = F.cosine_similarity(s2, t2, dim=-1).mean()
            cos_embedding.append(((cos1 + cos2) * 0.5).item())


    print(f"iter [{it}/{EPOCHS*len(train_data)}], loss:{np.mean(train_loss):.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}")
    torch.save({
        "iteration": it,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    }, "checkpoint.pth")
