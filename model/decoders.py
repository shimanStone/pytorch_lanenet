#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 13:53
# @Author  : shiman
# @File    : decoders.py
# @describe:

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FCNDecoder(nn.Module):
    def __init__(self, decode_layers, decode_channels=[512, 256], decode_last_stride=8):
        super(FCNDecoder, self).__init__()

        self._decode_channels = [512, 256]
        self._out_channel = 64
        self._decode_layers = decode_layers

        self._conv_layers = []
        for _ch in self._decode_channels:
            self._conv_layers.append(nn.Conv2d(_ch, self._out_channel, kernel_size=1, bias=False).to(device))

        self._conv_final = nn.Conv2d(self._out_channel, 2, kernel_size=1, bias=False)
        self._deconv = nn.ConvTranspose2d(self._out_channel, self._out_channel, kernel_size=4,
                                          stride=2,
                                          padding=1, bias=False)

        self._deconv_final = nn.ConvTranspose2d(self._out_channel, self._out_channel, kernel_size=16,
                                                stride=decode_last_stride,
                                                padding=4, bias=False)

    def forward(self, encode_data):
        ret = {}
        input_tensor = encode_data[self._decode_layers[0]]
        input_tensor.to(device)
        score = self._conv_layers[0](input_tensor)
        for i, layer in enumerate(self._decode_layers[1:]):
            deconv = self._deconv(score)
            input_tensor = encode_data[layer]
            score = self._conv_layers[i](input_tensor)

            fused = torch.add(deconv, score)
            score = fused

        deconv_final = self._deconv_final(score)
        score_final = self._conv_final(deconv_final)

        ret['logits'] = score_final
        ret['deconv'] = deconv_final

        return ret


