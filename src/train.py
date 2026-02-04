import torch
import torch.nn as nn
from data.dataset import MIADDataset
from torch.utils.data import DataLoader
from models.dinov2 import vit_small
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from functools import partial
from models.dinomaly import ViTillv2
from utils.loss import global_cosine_hm_percent
from utils.utils import get_gaussian_kernel, trunc_normal_, cal_anomaly_maps, compute_ad_metrics
from utils.optimizer import StableAdamW
from timm.scheduler import CosineLRScheduler
import numpy as np
from torch.nn import functional as F

# CONFIG FILE WITH PARAMETERS

class_list = ["catenary_dropper", "electrical_insulator", "metal_welding", "nut_and_bolt", "photovoltaic_module",
              "wind_turbine", "witness_mark"]

# class_list = ["wind_turbine"]

train_dataset = MIADDataset(dataset_path="miad", class_list=class_list, mode="train")

NUM_ITERATIONS = 10000
BATCH_SIZE = 2
EMBED_DIM = 384
NUM_HEADS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

encoder = vit_small()

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

model = ViTillv2(encoder=encoder, decoder=decoder, bottleneck=bottleneck, target_layers=target_layers)
model = model.to(DEVICE)

trainable_modules = nn.ModuleList([bottleneck, decoder])

for m in trainable_modules.modules():
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

optimizer = StableAdamW([{'params': trainable_modules.parameters()}],
                        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
lr_scheduler = CosineLRScheduler(optimizer, t_initial=NUM_ITERATIONS - 100, lr_min=2e-4, warmup_t=100, warmup_lr_init=0)

print("Starting training...")
it = 0

resize_mask = 256
max_ratio = 0.01  # top 1% of the pixels will be used for the anomaly score
for epoch in range(int(np.ceil(NUM_ITERATIONS / len(train_data)))):
    model.train()

    train_loss = []
    for batch in train_data:
        images, _, _ = batch
        images = images.to(DEVICE)

        encoded, decoded = model(images)

        p_final = 0.9
        p = min(p_final * it / 1000, p_final)
        loss = global_cosine_hm_percent(encoded, decoded, p=p)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trainable_modules.parameters(), max_norm=0.1)
        optimizer.step()

        lr_scheduler.step_update(it)

        train_loss.append(loss.item())


        # Evaluation...
        if (it + 1) % 1000 == 0:
            auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
            auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

            model.eval()
            for item in class_list:

                test_dataset = MIADDataset(dataset_path="miad", class_list=[item], mode="test")
                test_data = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

                gt_list_px = []
                pr_list_px = []
                gt_list_sp = []
                pr_list_sp = []

                with torch.no_grad():
                    for test_batch in test_data:
                        images, gt, labels = test_batch
                        images = images.to(DEVICE)
                        encoded, decoded = model(images)

                        anomaly_map, _ = cal_anomaly_maps(encoded, decoded, images.shape[-1])

                        if resize_mask is not None:
                            anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                            gt = F.interpolate(gt, size=resize_mask, mode='nearest')

                        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEVICE)

                        anomaly_map = gaussian_kernel(anomaly_map)

                        gt = gt.bool()
                        if gt.shape[1] > 1:
                            gt = torch.max(gt, dim=1, keepdim=True)[0]

                        gt_list_px.append(gt)
                        pr_list_px.append(anomaly_map)
                        gt_list_sp.append(labels)

                        if max_ratio == 0:
                            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
                        else:
                            anomaly_map = anomaly_map.flatten(1)
                            sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                                       :int(anomaly_map.shape[1] * max_ratio)]
                            sp_score = sp_score.mean(dim=1)
                        pr_list_sp.append(sp_score)
                        break

                    results = compute_ad_metrics(gt_list_px=gt_list_px, pr_list_px=pr_list_px, gt_list_sp=gt_list_sp, pr_list_sp=pr_list_sp)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print(
                        f"{item}: "
                        f"I-AUROC:{auroc_sp:.4f}, "
                        f"I-AP:{ap_sp:.4f}, "
                        f"I-F1:{f1_sp:.4f}, "
                        f"P-AUROC:{auroc_px:.4f}, "
                        f"P-AP:{ap_px:.4f}, "
                        f"P-F1:{f1_px:.4f}, "
                        f"P-AUPRO:{aupro_px:.4f}"
                    )

            print(
                f"Mean: "
                f"I-AUROC:{np.mean(auroc_sp_list):.4f}, "
                f"I-AP:{np.mean(ap_sp_list):.4f}, "
                f"I-F1:{np.mean(f1_sp_list):.4f}, "
                f"P-AUROC:{np.mean(auroc_px_list):.4f}, "
                f"P-AP:{np.mean(ap_px_list):.4f}, "
                f"P-F1:{np.mean(f1_px_list):.4f}, "
                f"P-AUPRO:{np.mean(aupro_px_list):.4f}"
            )

            model.train()

        it += 1
        print("One iterations done...")

    print(f"iter [{it}/{NUM_ITERATIONS}], loss:{np.mean(train_loss):.4f}")

