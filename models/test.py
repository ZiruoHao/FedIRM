#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os
import struct
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ColorMNIST(Dataset):
    def __init__(self, color, dataset, randomcolor=False):
        assert color in ['num', 'back', 'both'], "color must be either 'num', 'back' or 'both"
        if color=='both':
            self.pallette = [[31, 119, 180],
                            [255, 127, 14],
                            [44, 160, 44],
                            [214, 39, 40],
                            [148, 103, 189],
                            [140, 86, 75],
                            [227, 119, 194],
                            [127, 127, 127],
                            [188, 189, 34],
                            [23, 190, 207]]
        else:
            self.pallette = [[42, 19, 180],
                            [215, 10, 237],
                            [144, 60, 24],
                            [24, 139, 80],
                            [38, 173, 89],
                            [130, 26, 175],
                            [97, 112, 24],
                            [97, 27, 127],
                            [88, 209, 34],
                            [203, 90, 97]]
        self.color = color
        self.randomcolor = randomcolor
        self.dataset=dataset

    def __len__(self):
        # return len(self.dataset)
        return 1280
    

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image=np.array(image[0,:,:])*255
        image = np.tile(image[ :, :, np.newaxis], 3)
        if self.randomcolor:
            c = self.pallette[np.random.randint(0, 10)]
            if self.color == 'both':
                while True:
                    c2 = self.pallette[np.random.randint(0, 10)]
                    if c2 != c: break
        else:
            if self.color == 'num':
                # c = self.pallette[-(label + self.agent)%10]
                c = self.pallette[(9-label)%10]
            elif self.color == 'back':
                c = self.pallette[(9-label)%10]
            else:
                c = self.pallette[-label]
                c2 = self.pallette[-(label - 3)]

        # Assign color according to their class (0,10)
        # if self.color == 'num':
        #     image[0, :, :] = image[0, :, :] / 255 * c[0]
        #     image[1, :, :] = image[1, :, :] / 255 * c[1]
        #     image[2, :, :] = image[2, :, :] / 255 * c[2]
        # elif self.color == 'back':
        #     image[0, :, :] = (255 - image[0, :, :]) / 255 * c[0]
        #     image[1, :, :] = (255 - image[1, :, :]) / 255 * c[1]
        #     image[2, :, :] = (255 - image[2, :, :]) / 255 * c[2]
        # else:
        #     image[0, :, :] = image[0, :, :] / 255 * c[0] + (255 - image[0, :, :]) / 255 * c2[0]
        #     image[1, :, :] = image[:, :, 1] / 255 * c[1] + (255 - image[1, :, :]) / 255 * c2[1]
        #     image[2, :, :] = image[2, :, :] / 255 * c[2] + (255 - image[2, :, :]) / 255 * c2[2]
        if self.color == 'num':
            image[:, :, 0] = image[:, :, 0] / 255 * c[0]
            image[:, :, 1] = image[:, :, 1] / 255 * c[1]
            image[:, :, 2] = image[:, :, 2] / 255 * c[2]
        elif self.color == 'back':
            image[:, :, 0] = (255 - image[:, :, 0]) / 255 * c[0]
            image[:, :, 1] = (255 - image[:, :, 1]) / 255 * c[1]
            image[:, :, 2] = (255 - image[:, :, 2]) / 255 * c[2]
        else:
            image[:, :, 0] = image[:, :, 0] / 255 * c[0] + (255 - image[:, :, 0]) / 255 * c2[0]
            image[:, :, 1] = image[:, :, 1] / 255 * c[1] + (255 - image[:, :, 1]) / 255 * c2[1]
            image[:, :, 2] = image[:, :, 2] / 255 * c[2] + (255 - image[:, :, 2]) / 255 * c2[2]
        
        # image=array_to_img(image)
        image = Image.fromarray(np.uint8(image))
        image = transforms.ToTensor()(image)  # Range [0,1]

        return image, label

def test_img(net_g, datatest, args,colortype):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(ColorMNIST(colortype,datatest,randomcolor=False), batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        save_image(data,'test_img.jpg',nrow=10,padding=2)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss




# def test_img(net_g, datatest, args):
#     net_g.eval()
#     # testing
#     test_loss = 0
#     correct = 0
#     data_loader = DataLoader(datatest, batch_size=args.bs)
#     l = len(data_loader)
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.cuda(), target.cuda()
#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * correct / len(data_loader.dataset)
#     if args.verbose:
#         print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(data_loader.dataset), accuracy))
#     return accuracy, test_loss
