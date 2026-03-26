""""
Implementation from: https://github.com/guojiajeremy/Dinomaly/blob/master/models/uad.py
"""

import torch
import torch.nn as nn
import math


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)

        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
        de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

class ViTillSmall(ViTill):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
    ) -> None:
        super().__init__(encoder, bottleneck, decoder, target_layers)
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        """
        self.decoder_adapters = nn.ModuleList([
            nn.Linear(192, 384, bias=False)
            for _ in target_layers
        ])"""
        self.decoder_adapters = nn.ModuleList([
            nn.Conv2d(192, 384, kernel_size=1, bias=False)
            for _ in self.fuse_layer_decoder
        ])

    def forward(self, x):
        with torch.no_grad():
            en_list, en_raw = self.encoder(x)
        x = self.fuse_feature(en_raw)
        side = int(math.sqrt(en_raw[0].shape[1] - 1 - self.encoder.backbone.num_register_tokens))

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)
        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
        de = [d[:, 1 + self.encoder.backbone.num_register_tokens:, :] for d in de]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        de = [l(d) for d, l in zip(de, self.decoder_adapters)]
        return en_list, de



class ViTillSmallv2(ViTill):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
    ) -> None:
        super().__init__(encoder, bottleneck, decoder, target_layers)
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder

        self.decoder_adapters = nn.ModuleList([
            nn.Conv2d(192, 384, kernel_size=1, bias=False)
            for _ in self.fuse_layer_decoder
        ])

    def forward(self, x):
        with torch.no_grad():
            en_list, en_raw = self.encoder(x)
        x = self.fuse_feature(en_raw)
        side = int(math.sqrt(en_raw[0].shape[1] - 1 - self.encoder.backbone.num_register_tokens))

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)
        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        en = [e[:, 1 + self.encoder.backbone.num_register_tokens:, :] for e in en]
        de = [d[:, 1 + self.encoder.backbone.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        de = [l(d) for d, l in zip(de, self.decoder_adapters)]
        return en, de


class ViTillBSD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
    ) -> None:
        super(ViTillBSD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.patch_size = encoder.patch_size

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        B, C, img_h, img_w = x.shape
        h, w = img_h // self.patch_size, img_w // self.patch_size

        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
        de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([B, -1, h, w]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([B, -1, h, w]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

