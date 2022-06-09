#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 16:26
# @Author  : shiman
# @File    : dataloader.py
# @describe:


from torch.utils.data import Dataset
import cv2
import numpy as np
import random


class LaneDataSet(Dataset):

    def __init__(self, dataset, n_labels=5, transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform
        self.n_labels = n_labels

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip('\n').split(' ')
                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])
        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        self._shuffle()

    def _shuffle(self):
        c = list(zip(self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def _split_instance_gt(self, label_instance_img):
        """extract each label into separate binary channels"""
        no_of_instances = self.n_labels
        ins = np.zeros((no_of_instances, label_instance_img.shape[0], label_instance_img.shape[1]))
        label_unique = np.unique(label_instance_img)[1:]
        for _ch, label in enumerate(label_unique):
            ins[_ch, label_instance_img == label] = 1
        return ins

    def __getitem__(self, idx):
        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR)  # shape: h,w,c
        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.cv2.IMREAD_COLOR) # shape:h,w
        label_instance_img = cv2.imread(self._gt_label_instance_list[idx], cv2.IMREAD_UNCHANGED)

        if self.transform:
            img = self.transform(img)
            label_img = self.transform(label_img)
            label_instance_img = self.transform(label_instance_img)
        # instance channels separate binary channels
        # label_instance_img = self._split_instance_gt(label_instance_img)
        # reshape for pytorch
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        #
        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1

        return img, label_binary, label_instance_img


class Rescale(object):
    """
    output_size(height, width)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        h, w = output_size
        self.output_size = (w, h)

    def __call__(self, sample):
        sample = cv2.resize(sample, dsize=self.output_size, interpolation=cv2.INTER_NEAREST)
        return sample

