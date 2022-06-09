#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 14:34
# @Author  : shiman
# @File    : loss.py
# @describe:


from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
from torch.functional import F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=[0.5, 0.5], n_class=2, reduction='mean', device=device):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=1e-6, max=0.999999)
        target_onehot = torch.zeros((target.size(0), self.n_class, target.size(1),
                                    target.size(2))).to(self.device)
        loss = 0
        for i in range(self.n_class):
            target_onehot[:, i, ...][target == i] = 1
        for i in range(self.n_class):
            loss -= self.alpha[i]*(1-pt[:, i, ...])**self.gamma \
                    * target_onehot[:, i, ...]*torch.log(pt[:, i, ...])
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001):
        super(DiscriminativeLoss, self).__init__()

        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        assert self.norm in [1, 2]

    def forward(self, input, target):

        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, embedding, seg_gt):

        bs, embed_dim, h, w = embedding.size()
        embedding = embedding.reshape(bs, embed_dim, h*w)
        seg_gt = seg_gt.reshape(bs, h*w)

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(bs):
            embedding_b = embedding[b]  # shape: dim, h*w
            seg_gt_b = seg_gt[b]  # shape: h*w

            labels = torch.unique(seg_gt_b)
            labels = [i for i in labels if i != 0]
            num_lanes = len(labels)

            if num_lanes == 0:
                _nonsence = embedding_b.sum()
                _zero = torch.zeros_like(_nonsence)
                var_loss = var_loss + _nonsence*_zero
                dist_loss = dist_loss + _nonsence*_zero
                continue

            centroid_mean = []

            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue

                embedding_i = embedding_b*seg_mask_i
                mean_i = torch.sum(embedding_i, dim=1) / torch.sum(seg_mask_i)
                centroid_mean.append(mean_i)
                # var_loss
                var_loss = var_loss + torch.sum(F.relu(
                    torch.norm(embedding_i[:, seg_mask_i] - mean_i.reshape(embed_dim, 1),
                               dim=0) - self.delta_var) ** 2) / torch.sum(seg_mask_i) / num_lanes
            centroid_mean = torch.stack(centroid_mean)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1,1,embed_dim)
                centroid_mean2 = centroid_mean.reshape(1,-1,embed_dim)

                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_dist

                dist_loss =  dist_loss + torch.sum(F.relu(-dist + self.delta_dist) ** 2) / (
                        num_lanes * (num_lanes - 1)) / 2

        var_loss = var_loss / bs
        dist_loss = dist_loss / bs

        return var_loss, dist_loss