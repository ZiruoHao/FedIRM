#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
from time import sleep


from utils.sampling import mnist_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist
from models.Fed import FedAvg
from models.test import test_img
from progress.bar import IncrementalBar

if __name__ == '__main__':
    # parse args
    name='ERMnum2num'
    color='num'
    num_items=2000
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split user 
    if args.dataset == 'mnist':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trans_mnist = transforms.Compose([transforms.ToTensor()]) 
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users,num_items)
        else:
            exit('Error: only consider IID setting in FedIRM')
    else:
        exit('Error: unrecognized dataset')
    img_size = torch.Size([3, 28, 28])

    # build model
    if args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = [0 for i in range(args.epochs)]
    loss_test = [0 for i in range(args.epochs)]
    acc_train = [0 for i in range(args.epochs)]
    acc_test = [0 for i in range(args.epochs)]
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        
    for iter in range(args.epochs):
        loss_locals = []
        acc_locals=[]
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # for idx in tqdm(idxs_users):
        
        for idx in idxs_users:
            local = LocalUpdate(args=args,lr=args.lr, lam=args.lam, dataset=dataset_train, idxs=dict_users[idx],agent=idx,colortype=color)
            w, loss, acc = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))
            # sleep(0.01)
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        acc_test_, loss_test_ = test_img(net_glob, dataset_test, args,color)
        
        # loss_avg = sum(loss_locals) / len(loss_locals)
        
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_avg = sum(acc_locals) / len(acc_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        acc_train[iter]=acc_train[iter]+acc_avg
        loss_train[iter]=loss_train[iter]+loss_avg
        acc_test[iter]=loss_train[iter]+acc_test_
        loss_test[iter]=loss_test[iter]+loss_test_
        np.savetxt(name+'/'+'loss_train.txt',loss_train)
        np.savetxt(name+'/'+'loss_test.txt',loss_test)
        np.savetxt(name+'/'+'acc_train.txt',acc_train)
        np.savetxt(name+'/'+'acc_test.txt',acc_test)

    # testing
    net_glob.eval()
    # acc_train, loss_train = test_img_back(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args,color)
    print("Training accuracy: {:.2f}".format(acc_avg))
    print("Testing accuracy: {:.2f}".format(acc_test))

