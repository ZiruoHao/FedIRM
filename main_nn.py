#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random
from utils.options import args_parser
from models.Nets import MLP
from utils.sampling import mnist_iid
from models.test import test_img
from PIL import Image


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

    
def penalty(logits, y):
    loss_func = torch.nn.CrossEntropyLoss()
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = loss_func(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    penalty_regular=torch.sum(grad**2)
    return penalty_regular

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)
    name='C-IRMback2back'
    color='back'
    repeattimes=10
    num_items=2000
    acctestlist=[0 for i in range(repeattimes)]
    acc_test = [0 for i in range(args.epochs)]
    # load dataset and split users
    if args.dataset == 'mnist':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trans_mnist = transforms.Compose([transforms.ToTensor()]) 
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users,num_items)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = torch.Size([3, 28, 28])
    # build model
    
    for repeat in range(repeattimes):
        if args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')
        print(net_glob)
        # training
        
        optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
        
        
        list_loss = []
        net_glob.train()
        train_loader=[0 for i in range(args.num_users)]
        for idx in range(args.num_users):
            train_loader[idx]= DataLoader(DatasetSplit(dataset_train, idxs=dict_users[idx],color=color,randomcolor=False,agent=idx, error=args.error), batch_size=args.local_bs, shuffle=True)
        # train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
        for epoch in range(args.epochs):
            batch_loss = []
            for agent in range(args.num_users):
                for batch_idx, (data, target) in enumerate(train_loader[agent]):
                    data, target = data.to(args.device), target.to(args.device)
                    optimizer.zero_grad()
                    output = net_glob(data)
                    loss = F.cross_entropy(output, target)
                    if epoch<10:
                        loss=loss
                    elif args.lam<1 :
                        loss = loss+ args.lam*penalty(output, target)
                        # print('penalty')
                    else:
                        loss=loss/ args.lam+penalty(output , target)
                        # print('penalty')
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
            
                
            acc_test_, loss_test_ = test_img(net_glob, dataset_test, args,color)
            acc_test[epoch]=acc_test[epoch]+float(acc_test_)
            np.savetxt(name+'/'+'acc_test.txt',acc_test)
            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)
            list_loss.append(loss_avg)
            
        net_glob.eval()
        acc_test1, loss_test_ = test_img(net_glob, dataset_test, args,color)
        print("Testing accuracy: {:.2f}".format(acc_test1))
        acctestlist[repeat]=acc_test1

    for epoch in range(args.epochs):
        acc_test[epoch]=acc_test[epoch]/repeattimes
        
    np.savetxt(name+'/'+'acc_test.txt',acc_test)
    np.savetxt(name+'/'+'acclist.txt',acctestlist)