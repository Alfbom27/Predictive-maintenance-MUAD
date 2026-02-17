import torch
import torch.nn as nn
from models.dinov2 import vit_tiny, vit_base
from data.dataset import KDdataset, MIADDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.student import Student, StudentHead
from models.teacher import Teacher
from utils.utils import WarmCosineScheduler
import numpy as np
import random
import torch.nn.functional as F

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


class_list = ["electrical_insulator", "metal_welding", "photovoltaic_module", "wind_turbine"]

miad_data = MIADDataset(dataset_path="miad", class_list=class_list, mode="train", kd_training=True)
train_dataset = KDdataset(miad_data)

EPOCHS = 100
WARMUP_EPOCHS = 10
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
#                        pin_memory=True, collate_fn=multi_collate, persistent_workers=True, worker_init_fn=seed_worker)

train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=multi_collate)


# Student

K = 1000

student_backbone = vit_tiny()
student_head = StudentHead(in_dim=192, out_dim=768)
student = Student(backbone=student_backbone, head=student_head)

# Teacher
teacher_backbone = vit_base(
    patch_size=14,
    img_size=518,
    block_chunks=0,
    init_values=1e-8,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
)
ckpt = torch.load(
    "../PycharmProjects/Pythonprojects/Predictive-maintenance-MUAD/src/weights/dinov2_vitb14_pretrain.pth",
    map_location="cpu")
teacher_backbone.load_state_dict(ckpt, strict=True)

teacher = Teacher(backbone=teacher_backbone)

for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()

# Optim and Lr scheduler

# TODO: add weight decay cosine scheduler
optimizer = AdamW(student.parameters(), lr=2e-3, weight_decay=4e-2)

lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=1e-6, total_iters=EPOCHS * len(train_data),
                                   warmup_iters=WARMUP_EPOCHS * len(train_data))

it = 0
for epoch in range(EPOCHS):
    train_loss = []
    cos_embedding = []

    student.train()
    for batch in train_data:
        global_crops = batch[:2]
        local_crops = batch[2:]
        B = global_crops[0].shape[0]

        # teacher pred
        with torch.no_grad():
            teacher_global = teacher(torch.cat(global_crops, dim=0))
            teacher_global = F.normalize(teacher_global, dim=-1)


        t1, t2 = teacher_global[:B], teacher_global[B:]

        student_global = student(torch.cat(global_crops, dim=0))
        student_global = F.normalize(student_global, dim=-1)

        student_local = student(torch.cat(local_crops, dim=0))
        student_local = F.normalize(student_local, dim=-1)

        s1, s2 = student_global[:B], student_global[B:]
        s_locals = student_local.chunk(len(local_crops)) # List of local crops over batch

        print(t1.shape, t2.shape)
        print(s1.shape, s2.shape)
        print(len(s_locals))

        # Loss
        num_views = 2 + len(s_locals)
        loss = 0

        loss += cosine_loss(s1, t1)
        loss += cosine_loss(s2, t2)

        for s_v in s_locals:
            loss += 0.5 * (cosine_loss(s_v, t1) + cosine_loss(s_v, t2))

        loss /= num_views


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        train_loss.append(loss.item())
        with torch.no_grad():
            cos1 = F.cosine_similarity(s1, t1, dim=-1).mean()
            cos2 = F.cosine_similarity(s2, t2, dim=-1).mean()
            cos_embedding.append(((cos1 + cos2) * 0.5).item())

        it += 1
    print(f"iter [{it}/{EPOCHS*len(train_data)}], loss:{np.mean(train_loss):.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}")

    break
