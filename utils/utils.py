#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 18:25
# @Author  : shiman
# @File    : utils.py
# @describe:

import argparse


class AverageMeter():
    """computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset path', default='E:\ml_code\pytorch_lanenet\data')
    parser.add_argument('--save', required=False, help='Directory of save model checkpoint', default=r'E:\ml_code\ytorch_lanenet\data\checkpoints')
    parser.add_argument('--epochs', required=False, type=int, help='Training epochs', default=10)
    parser.add_argument('--bs', required=False, type=int, help='batch size', default=16)
    parser.add_argument('--val', required=False, type=bool, help='use validation', default=True)
    parser.add_argument('--lr', required=False, type=float, help='learning rate', default=5e-5)
    parser.add_argument('--pretrained', required=False, default=None, help='pretrained model path')
    parser.add_argument('--image', default='E:\ml_code\lytorch_lanenet\data\output', help='output image folder')
    parser.add_argument('--net', help='backbone network')
    parser.add_argument('--json', help='post processing json')
    return parser.parse_args()