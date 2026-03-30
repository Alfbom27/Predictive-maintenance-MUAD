import torch
import torch.nn as nn
import timm


class DeiT(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        backbone =timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=0, img_size=img_size)

        self.backbone = backbone
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        self.blocks = backbone.blocks
        self.norm = backbone.norm

        self.num_register_tokens = 0

    def prepare_tokens(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:]
        cls_token = self.cls_token + self.pos_embed[:, :1]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x)
        return x