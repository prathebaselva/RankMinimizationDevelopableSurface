import torch.utils.data as data
import numpy as np
import math
import torch
import os
from scipy.spatial import cKDTree
from numpy.random import rand, seed, shuffle
#from gtSDF import getSDF



class gridData(data.Dataset):
    def __init__(self, grid_N=128):
        self.max_dimensions = np.ones((3, ))
        self.min_dimensions = -np.ones((3, ))

        seed()

        bounding_box_dimensions = self.max_dimensions - self.min_dimensions  # compute the bounding box dimensions of the point cloud
        #print("bounding box dim = ",bounding_box_dimensions)
        #grid_spacing = max(bounding_box_dimensions) / (args.grid_N - 9)  # each cell in the grid will have the same size
        #dim = np.arange(self.min_dimensions[0] - grid_spacing*4, self.max_dimensions[0] + grid_spacing*4, grid_spacing)
        grid_spacing = max(bounding_box_dimensions) / grid_N  # each cell in the grid will have the same size
        dim = np.arange(self.min_dimensions[0] - grid_spacing, self.max_dimensions[0] + grid_spacing, grid_spacing)
        X, Y, Z = np.meshgrid(list(dim), list(dim), list(dim))

        self.grid_shape = X.shape
        self.batchsize = 4096 
        self.samples_xyz = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
        self.number_samples = self.samples_xyz.shape[0]
        self.xyz = self.samples_xyz
        #import trimesh
        #trimesh.Trimesh(vertices=self.xyz).export('grid.ply')
        #exit()
        self.number_batches = math.ceil(self.number_samples * 1.0 / self.batchsize)

    def setBatchSize(self, batchsize):
        self.batchsize = batchsize
   
    def __len__(self):
        self.number_batches = math.ceil(len(self.xyz) * 1.0 / self.batchsize)
        return self.number_batches

    def __getitem__(self, idx):
        print(idx, flush=True)
        start_idx = idx * self.batchsize
        end_idx = min(start_idx + self.batchsize, self.number_samples)  # exclusive
        #print("number samples = ",self.number_samples)
        this_bs = end_idx - start_idx
        end_idx = min(start_idx + self.batchsize, self.number_samples)  # exclusive
        xyz = torch.tensor(self.samples_xyz[start_idx:end_idx, :])

        return torch.FloatTensor(xyz.float())


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description = "Get gtSDF")
    parser.add_argument("-f","--filename", help="Input file name", required=True)
    args = parser.parse_args()
    #filepath = open(args.filename, 'r')
    filepath = np.load(args.filename, allow_pickle=True)
    for line in filepath:
        filename = line.strip()
        print(filename, flush=True)
        dataset = DeepSdfDataset(fnames=filename)
    filepath.close()
    #dataset = DeepSdfDataset(fnames=filename)

