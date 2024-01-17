import numpy as np
import random
import os
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


def initOptimizer(model, latent, args):
    if args.optimizer == 'adam':
        optimizer = Adam([{"params":filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr}, {"params":latent.parameters(), "lr":args.latlr}], weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    return optimizer


def initNoLatentOptimizer(model):
    optimizer = Adam([{"params":filter(lambda p: p.requires_grad, model.parameters()), "lr": 1e-5}], weight_decay=0.01)
    print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    return optimizer

def initScheduler(optimizer):
    scheduler = ReduceLROnPlateau(optimizer, factor=0.99 ,patience=10,mode='min',threshold=1e-4, eps=0, min_lr=0)
    return scheduler

