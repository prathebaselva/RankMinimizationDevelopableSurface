#import open3d
import torch.backends.cudnn as cudnn
from initialize import initNoLatentOptimizer, initScheduler
from model import Model
from dataset import gridData
import torch
import trimesh
import os
import math
import random
import numpy as np
from loguru import logger

outfolder = 'output/'

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)
   

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Tester(object):
    def __init__(self, model_path,grid_N,  device):
        self.device = device
        cudnn.benchmark = True
        self.initialize(model_path, grid_N)
        self.load_pretrained_checkpoint(model_path)
   
    def initialize(self, model_path, grid_N):
        self.model = Model(model_path, self.device).to(self.device)
        self.optimizer = initNoLatentOptimizer(self.model.net)
        self.scheduler = initScheduler(self.optimizer)
        self.grid_samples = gridData(grid_N)

    def load_pretrained_checkpoint(self, model_path):
        map_location = {'cuda:%d' %0: 'cuda:%d' % 0}

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'optimizer' in checkpoint:
                print("optimizer")
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                print("scheduler")
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            #if 'lr' in checkpoint:
            #    self.curr_lr = checkpoint['lr']
            logger.info(f'Training resumes from pretrained {model_path}')
        else:
            logger.info(f'{model_path} not found ')
