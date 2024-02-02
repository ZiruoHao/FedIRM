#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import numpy as np
import random
from sklearn import metrics
import os
import struct
import torch
from PIL import Image
from torchvision import transforms
# from keras.preprocessing.image import array_to_img



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, color,  randomcolor=False,agent=None, error=None):
        self.dataset = dataset
        self.agent=agent
        self.idxs = list(idxs)
        assert color in ['num', 'back', 'both'], "color must be either 'num', 'back' or 'both"
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
        self.randomcolor = randomcolor
        self.color = color
        self.error=error
        self.rantrans=int(self.agent/10+1)


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # err=random.random()
        # if err<self.error:
        label=round((label+random.gauss(0,self.error)))%10   

            # label=(label+1)%10

        # print(type(image))
        # image = torch.cat((image,image,image), dim=3, out=None)
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
                c = self.pallette[(label+self.agent)%10]
            elif self.color == 'back':
                c = self.pallette[(label+self.agent)%10]
            else:
                c = self.pallette[(label+self.agent)%10]
                c2 = self.pallette[(label+self.agent+self.rantrans)%10]

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


class LocalUpdate(object):
    def __init__(self, args, lr,lam ,dataset=None, idxs=None,agent=None,colortype=None):
        self.args = args
        self.lam=lam
        self.lr=lr
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.agent=agent
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, colortype,randomcolor=False,agent=self.agent, error=self.args.error), batch_size=self.args.local_bs, shuffle=True)
        
    # def mean_nll(self,logits, y):
    #     return nn.CrossEntropyLoss(logits, y)
    
    def penalty(self,logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = self.loss_func(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        penalty_regular=torch.sum(grad**2)
        return penalty_regular

    def train(self, net):
        # scaler = GradScaler()
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)

        epoch_loss = []
        # print(len(self.ldr_train.dataset))
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                correct = 0
                batch_loss = []
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                save_image(images,'train_img'+str(self.agent)+'.jpg',nrow=10,padding=2)
                net.zero_grad()
                with autocast():
                    log_probs = net(images)
                    if self.lam<1:
                        loss = self.loss_func(log_probs, labels)+ self.lam*self.penalty(log_probs, labels)
                    else:
                        loss=self.loss_func(log_probs, labels)/ self.lam+self.penalty(log_probs, labels)
                loss.backward()
                optimizer.step()     
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct +=  y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                # print(correct)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        accuracy = 100.00 * correct / len(self.ldr_train.dataset)
        # print(correct.item())
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), accuracy

