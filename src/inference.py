import torch
import torch.nn as nn

from models.dinov2 import vit_small, vit_tiny
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from functools import partial
from models.dinomaly import ViTillv2, ViTillSmall, ViTillv2Small
from data.dataset import MIADDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.student import StudentFKD
from utils.utils import cal_anomaly_maps, get_gaussian_kernel
from matplotlib import pyplot as plt
import numpy as np


def visualize(images, gts, anomaly_maps, max_cols=10):
    images = images.detach().cpu()
    gts = gts.detach().cpu()
    anomaly_maps = anomaly_maps.detach().cpu()

    B = images.shape[0]
    cols = min(B, max_cols)

    fig, axes = plt.subplots(3, cols, figsize=(3 * cols, 9))

    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(cols):
        img = images[i]
        img = img.permute(1, 2, 0).numpy()

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis("off")

        gt = gts[i].squeeze().numpy()
        axes[1, i].imshow(gt, cmap="gray")
        axes[1, i].set_title("GT")
        axes[1, i].axis("off")


        amap = anomaly_maps[i].squeeze().numpy()
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

        axes[2, i].imshow(amap, cmap="jet")
        axes[2, i].set_title("Anomaly")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.show()



# EMBED_DIM = 384
# NUM_HEADS = 6
# EMBED_DIM = 192
# NUM_HEADS = 3
EMBED_DIM = 384
NUM_HEADS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

"""
encoder = vit_small(
    patch_size=14,
    img_size=518,
    block_chunks=0,
    init_values=1e-8,
    num_register_tokens=4,
    interpolate_antialias=False,
    interpolate_offset=0.1,
)
"""

vit_t = vit_tiny(
    patch_size=14,
    img_size=518,
    block_chunks=0,
    init_values=1e-8,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
)

encoder = StudentFKD(backbone=vit_t, teacher_dims=384, student_dims=192)

# ckpt = torch.load("../PycharmProjects/Pythonprojects/Predictive-maintenance-MUAD/src/weights/dinov2_vits14_reg4_pretrain.pth", map_location="cpu")
# encoder.load_state_dict(ckpt, strict=True)
# ckpt = torch.load("../PycharmProjects/Pythonprojects/Predictive-maintenance-MUAD/src/weights/checkpoint_vit_tiny_encoder_3k.pth", map_location="cpu")
# encoder.load_state_dict(ckpt["model_state_dict"], strict=True)

for p in encoder.parameters():
    p.requires_grad = False

bottleneck = []
decoder = []

bottleneck.append(bMlp(EMBED_DIM, EMBED_DIM * 4, EMBED_DIM, drop=0.2))
bottleneck = nn.ModuleList(bottleneck)

for i in range(8):
    blk = VitBlock(dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=4.,
                   qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                   attn=LinearAttention2)

    decoder.append(blk)

decoder = nn.ModuleList(decoder)

model = ViTillSmall(encoder=encoder, decoder=decoder, bottleneck=bottleneck, target_layers=target_layers)

checkpoint = torch.load("../PycharmProjects/Pythonprojects/Predictive-maintenance-MUAD/src/weights/dinomaly_ckpt/vitill_tiny_kd/checkpoint_miad_vit_tiny_70k.pth", map_location=DEVICE, weights_only=False)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()

# Data

# class_list = ["electrical_insulator", "metal_welding", "photovoltaic_module", "wind_turbine"]

class_list = ["wind_turbine"]

SAMPLE_SIZE = 5

test_dataset = MIADDataset(dataset_path="miad", class_list=class_list, mode="test")

test_data = DataLoader(dataset=test_dataset, batch_size=SAMPLE_SIZE, shuffle=True, drop_last=True)

resize_mask = 256
gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEVICE)
with torch.no_grad():
    for batch in test_data:
        images, gt, labels = batch

        images = images.to(DEVICE)
        encoded, decoded = model(images)

        anomaly_maps, _ = cal_anomaly_maps(encoded, decoded, images.shape[-1])

        if resize_mask is not None:
            anomaly_map = F.interpolate(anomaly_maps, size=resize_mask, mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=resize_mask, mode='nearest')

        anomaly_maps = gaussian_kernel(anomaly_maps)

        visualize(images, gt, anomaly_maps)
        break