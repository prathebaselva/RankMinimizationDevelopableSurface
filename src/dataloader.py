import torch.utils.data as data
import numpy as np
import math
import torch
import os
from scipy.spatial import cKDTree
from numpy.random import rand, seed, shuffle
from utils import *
from gtSDF import *



class SdfDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, args=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.numfiles = len(dataset)
        self.num_workers= 1
   
    def __len__(self):
        return self.files

    def __getitem__(self, idx):
        start_idx = idx * self.batchsize
        end_idx = min(start_idx + self.batchsize, self.number_samples)  # exclusive
        #print("number samples = ",self.number_samples)
        this_bs = end_idx - start_idx
        assert self.phase == 'test'
        end_idx = min(start_idx + self.batchsize, self.number_samples)  # exclusive
        xyz = torch.tensor(self.samples_xyz[start_idx:end_idx, :])

        return torch.FloatTensor(xyz.float())
