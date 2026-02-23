# Implementaiton from: https://github.com/guojiajeremy/Dinomaly/blob/master/utils.py

import torch
import torch.nn as nn
from functools import partial

def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x

def global_cosine_hm_percent(a, b, p=0.9, factor=0.1):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def dino_loss_ce(s_logits, t_logits, T_s=0.1, T_t=0.07):
    with torch.no_grad():
        p_t = torch.softmax(t_logits / T_t, dim=-1)

    log_p_s = torch.log_softmax(s_logits / T_s, dim=-1)
    return -(p_t * log_p_s).sum(dim=-1).mean()

class RoBLoss(nn.Module):
    def __init__(self, T_s=0.1, T_t=0.07, lambda_patch=1.0):
        super().__init__()
        self.T_s = T_s
        self.T_t = T_t
        self.lambda_patch = lambda_patch

    def global_loss(self, s_cls, t_cls):
        return dino_loss_ce(s_cls, t_cls, self.T_s, self.T_t)

    def patch_loss(self, s_patch, t_patch, mask):
        return dino_loss_ce(
            s_patch[mask],
            t_patch[mask],
            self.T_s,
            self.T_t
        )

    def forward(self, s_cls, t_cls, s_patch=None, t_patch=None, mask=None):
        loss = self.global_loss(s_cls, t_cls)

        if s_patch is not None:
            loss += self.lambda_patch * self.patch_loss(s_patch, t_patch, mask)

        return loss
