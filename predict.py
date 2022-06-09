#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 16:00
# @Author  : shiman
# @File    : predict.py
# @describe:


import cv2
import numpy as np
from PIL import Image

from torch.autograd import Variable
import torch
from torchvision import transforms
from pytorch_lanenet.model.dataloader import Rescale
from pytorch_lanenet.model.model import LaneNet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':

    model_path = r'E:\ml_code\pytorch_lanenet\data\rst\9_checkpoint.pth'

    image_path = r'E:\data\lanenet\testing\gt_image\0000.png'
    resize_height, resize_width = 256, 512

    data_transform = transforms.Compose([
        Rescale((resize_height, resize_width)),
        transforms.ToTensor()
    ])

    # model = LaneNet()
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict)
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)

    dummy_input = cv2.imread(image_path, cv2.IMREAD_COLOR)
    dummy_input =  cv2.resize(dummy_input, dsize=(resize_width, resize_height), interpolation=cv2.INTER_NEAREST)
    dummy_input = np.expand_dims(np.transpose(dummy_input, (2,0,1)), axis=0)
    dummy_input = torch.from_numpy(dummy_input)

    dummy_input = Variable(dummy_input).type(torch.FloatTensor).to(device)

    outputs = model(dummy_input)


    input = Image.open((image_path))
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy()*255
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy()*255

    out_root = r'E:\ml_code\pytorch_lanenet\data\pred'
    cv2.imwrite(f'{out_root}/input.jpg', input)
    cv2.imwrite(f'{out_root}/instance_output.jpg', instance_pred.transpose((1,2,0)))
    cv2.imwrite(f'{out_root}/binary_output.jpg', binary_pred)

    print('ok')
