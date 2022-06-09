#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 10:31
# @Author  : shiman
# @File    : encoders.py
# @describe:

import torch
import torch.nn as nn
from collections import OrderedDict

import torchvision.models as models


class VGGEncoder(nn.Module):

    def __init__(self, num_blocks, in_channels, out_channels):
        super(VGGEncoder, self).__init__()

        self.pretrained_modules = models.vgg16(pretrained=True).features

        self.num_blocks = num_blocks
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._conv_reps = [2, 2, 3, 3, 3]  # 13*2(conv+relu)+5(maxpool)
        self.net = nn.Sequential()
        self.pretrained_net = nn.Sequential()

        for i in range(num_blocks):
            self.net.add_module(f'block_{i+1}', self._encode_block(i+1))
            self.pretrained_net.add_module(f'block_{i+1}', self._encode_pretrained_block(i+1))

    def _encode_block(self, block_id, kernel_size=3, stride=1):
        out_channels = self._out_channels[block_id-1]
        padding = (kernel_size-1)//2
        seq = nn.Sequential()

        for i in range(self._conv_reps[block_id-1]):
            if i == 0:
                in_channels = self._in_channels[block_id-1]
            else:
                in_channels = out_channels
            seq.add_module(f'conv_{block_id}_{i+1}',
                           nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding))
            seq.add_module(f'bn_{block_id}_{i+1}',
                           nn.BatchNorm2d(out_channels))
            seq.add_module(f'relu_{block_id}_{i+1}', nn.ReLU())
        seq.add_module(f'maxpool{block_id}', nn.MaxPool2d(kernel_size=2, stride=2))
        return seq

    def _encode_pretrained_block(self, block_id):
        seq = nn.Sequential()
        for i in range(0, self._conv_reps[block_id-1], 4):
            seq.add_module(f'conv_{block_id}_{i+1}', self.pretrained_modules[i])
            seq.add_module(f'relu_{block_id}_{i+2}', self.pretrained_modules[i+1])
            seq.add_module(f'conv_{block_id}_{i+3}', self.pretrained_modules[i+2])
            seq.add_module(f'relu_{block_id}_{i+4}', self.pretrained_modules[i+3])
            seq.add_module(f'maxpool{block_id}', self.pretrained_modules[i+4])
        return seq

    def forward(self, input_tensor):
        ret = OrderedDict()
        X = input_tensor
        for i, block in enumerate(self.net):
            # print(X.size())
            pool = block(X)
            ret[f'pool{i+1}'] = pool
            X = pool
        return ret








