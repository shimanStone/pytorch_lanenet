#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 14:34
# @Author  : shiman
# @File    : model.py
# @describe:

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import DiscriminativeLoss

from .encoders import VGGEncoder
from .decoders import FCNDecoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LaneNet(nn.Module):
    def __init__(self, arch='VGG'):
        super(LaneNet, self).__init__()

        self.no_of_instances = 5
        encode_num_blocks = 5
        in_channels = [3, 64, 128, 256, 512]
        out_channels = [64, 128, 256, 512, 512]
        self._arch = arch
        if self._arch == 'VGG':
            self._encoder = VGGEncoder(encode_num_blocks, in_channels, out_channels)
            self._encoder.to(device)

            decode_layers = ['pool5', 'pool4', 'pool3']
            decode_channels = out_channels[:-(len(decode_layers)+1):-1]
            decode_last_stride = 8

            self._decoder = FCNDecoder(decode_layers, decode_channels, decode_last_stride)
            self._decoder.to(device)
        else:
            raise NotImplementedError

        self._pix_layer = nn.Conv2d(in_channels=64, out_channels=self.no_of_instances, kernel_size=1,
                                    bias=False).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, input_tensor):
        encode_ret = self._encoder(input_tensor)
        decode_ret = self._decoder(encode_ret)

        decode_logits = decode_ret['logits']
        decode_logits = decode_logits.to(device)

        binary_seg_ret = torch.argmax(F.softmax(decode_logits, dim=1), dim=1, keepdim=True)

        decode_deconv = decode_ret['deconv']

        pix_embedding = self.relu(self._pix_layer(decode_deconv))

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': decode_logits
        }


def compute_loss(net_output, binary_label, instance_label):
    k_binary = 0.7
    k_instance = 0.3
    k_dist = 1.0

    ce_loss_fn = nn.CrossEntropyLoss()
    binary_seg_logits = net_output['binary_seg_logits']
    binary_loss = ce_loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output['instance_seg_logits']
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    instance_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    total_loss = binary_loss + instance_loss + dist_loss
    out = net_output['binary_seg_pred']
    # iou = 0
    # batch_size = out.size()[0]
    # for i in range(batch_size):
    #     PR = out[i].squeeze(0).nonzero().size()[0]
    #     GT = binary_label[i].nonzero().size()[0]
    #     TP = (out[i].sequeeze(0)*binary_label[i]).nonzero().size()[0]
    #     union = PR + GT - TP
    #     iou += TP / union
    # iou = iou / batch_size
    return total_loss, binary_loss, instance_loss, dist_loss,out
