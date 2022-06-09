#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 17:30
# @Author  : shiman
# @File    : train.py
# @describe:


import time
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)
print(f'cur_dir: {cur_dir}')

sys.path.append(os.path.abspath(cur_dir))
sys.path.append(os.path.abspath(root_dir))
print(sys.path)

from tqdm import tqdm

import torch
from pytorch_lanenet.model.dataloader import LaneDataSet
from pytorch_lanenet.model.dataloader import Rescale
from pytorch_lanenet.model.model import LaneNet, compute_loss

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

from pytorch_lanenet.utils.utils import parse_args, AverageMeter


vgg_mean = [103.939, 116.779, 123.68]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compose_img(image_data, out, binary_label, pix_embedding, instance_label, i):
    pass


def train(train_loader, model, optimizer, total_epochs, epoch):
    # mean_iou = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{total_epochs}',
              postfix=dict, mininterval=0.3) as pbar:
        for iter, batch in enumerate(train_loader):
            if iter > len(train_loader):
                break

            image_data = Variable(batch[0]).type(torch.FloatTensor).to(device)
            binary_label = Variable(batch[1]).type(torch.LongTensor).to(device)
            instance_label = Variable(batch[2]).type(torch.FloatTensor).to(device)
            # forward
            net_output = model(image_data)
            # loss
            total_loss, binary_loss, instance_loss, out = compute_loss(net_output, binary_label, instance_label)
            # update loss
            total_losses.update(total_loss.item(), image_data.size()[0])
            binary_losses.update(binary_loss.item(), image_data.size()[0])
            instance_losses.update(instance_loss.item(), image_data.size()[0])
            # mean_iou.update(train_iou, image_data.size()[0])
            #
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'total_loss': total_losses.avg,
                                'bin_loss': binary_losses.avg,
                                'ins_loss': instance_losses.avg,})
                                # 'iou': train_iou
            pbar.update(1)

            if iter % 500 == 0:
                train_img_list = []

    # return mean_iou.avg
    return None


def test(val_loader, model, total_epochs, epoch):
    model.eval()
    mean_iou = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()

    with tqdm(total=len(val_loader), desc=f'Epoch {epoch}/{total_epochs}',
              postfix=dict, mininterval=0.3) as pbar:
        for iter, batch in enumerate(val_loader):
            if iter > len(val_loader):
                break

            image_data = Variable(batch[0]).type(torch.FloatTensor).to(device)
            binary_label = Variable(batch[1]).type(torch.FloatTensor).to(device)
            instance_label = Variable(batch[2]).type(torch.FloatTensor).to(device)
            # forward
            net_output = model(image_data)
            # loss
            total_loss, binary_loss, instance_loss, out = compute_loss(net_output, binary_label, instance_label)
            # update loss
            total_losses.update(total_loss.item(), image_data.size()[0])
            binary_losses.update(binary_loss.item(), image_data.size()[0])
            instance_losses.update(instance_loss.item(), image_data.size()[0])
            # mean_iou.update(train_iou, image_data.size()[0])
            #

            pbar.set_postfix(**{'total_loss': total_losses.avg,
                                'bin_loss': binary_losses.avg,
                                'ins_loss': instance_losses.avg})
                                # 'iou': train_iou})

            pbar.update(1)

    # return mean_iou.avg
    return None


def save_model(save_path, epoch, model):
    save_name = f'{save_path}/{epoch}_checkpoint.pth'
    torch.save(model, save_name)
    print(f'model is saved: {save_name}')


def main():
    args = parse_args()
    # model save path
    save_path = args.save
    os.makedirs(save_path, exist_ok=True)
    # dataset_file.txt
    train_dataset_file = f'{args.dataset}/train.txt'
    val_dataset_file = f'{args.dataset}/val.txt'
    # dataloader
    transform = transforms.Compose([Rescale((256, 512))])
    train_dataset = LaneDataSet(train_dataset_file, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    if args.val:
        val_dataset = LaneDataSet(val_dataset_file, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    model = LaneNet()
    model.to(device)
    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'Total || {args.epochs} Epochs, {len(train_dataset)} Training samples\n')

    for epoch in range(0, args.epochs):
        print(f'Epoch {epoch}')
        train_iou = train(train_loader, model, optimizer, args.epochs, epoch)
        # print(f'train iou: {train_iou}')

        if args.val:
            val_iou = test(val_loader, model, args.epochs, epoch)
            # print(f'val iou: {val_iou}')

        if (epoch+1) % 5 == 0:
            save_model(save_path, epoch, model)


if __name__ == '__main__':
    main()








