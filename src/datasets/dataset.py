import torch.utils.data as data
import numpy as np
import math
import torch
import os
from scipy.spatial import cKDTree
from numpy.random import rand, seed, shuffle
from src.utils.utils import convertToPLY, clusterByNormal
from .gtSDF import getSDF
#from gtSDF import getSDF

ROOT_DIR = '/work/pselvaraju_umass_edu/Project_DevelopableSurface/DevelopableSurface/'
#datadir = os.path.join(ROOT_DIR,'data/250k_sampled/')
gtdir = os.path.join(ROOT_DIR,'data/gtSDF/')


class DeepSdfSingleDataset(data.Dataset):
    def __init__(self, fname, pointsnormals, args=None, phase='train'):
        self.phase = phase
        self.filename = fname
        self.pointsnormals = pointsnormals
        self.numpoints = len(self.pointsnormals)
        self.subsample = args.subsample
        #self.datadir = os.path.join(ROOT_DIR, args.datadir)
        self.device = 'cuda'
        self.gtdir = gtdir #gtdir #  os.patj.join(ROOT_DIR, args.gtdir)
        self.args = args
        #print(self.filenames)
        self.initialize()
        self.data = np.load(self.getSDF(self.filename), allow_pickle=True).item()
        #self.cluster = np.load(self.getCluster(), allow_pickle=True).item()

    def __len__(self):
        return self.numpoints

    def initialize(self):
        fname = self.filename
        gtfile = os.path.join(self.gtdir, fname+'_'+self.phase+'.npy')
        if not os.path.exists(gtfile):
            gtdata = self.getSDF(fname)
            np.save(gtfile, gtdata)

    def getCluster(self):
        fname = self.filename
        clusterpath = os.path.join(self.gtdir, fname+'_cluster_'+self.phase+'.npy')
        if not os.path.exists(clusterpath):
            clusterByNormal(clusterpath, self.pointsnormals[:,:3], self.pointsnormals[:,3:])
        return clusterpath

    def getSDF(self, fname):
        gtfile = os.path.join(self.gtdir, fname+'_'+self.phase+'.npy')
        if os.path.exists(gtfile):
            return gtfile

        #convertToPLY(self.pointsnormals[:,0:3], self.pointsnormals[:,3:], isVal = (self.phase == 'val'), fname=self.filename)
        #exit()
        points = torch.Tensor(self.pointsnormals[:,0:3])
        normals = torch.Tensor(self.pointsnormals[:,3:])
        number_points = points.shape[0]
              
        pert_number_points = int(len(points)/2)
        rand_number_points = int(0.1* len(points))

        samples_indices1 = np.random.randint(len(points), size=pert_number_points,)
        points1 = points[samples_indices1, :]
        normals1 = normals[samples_indices1, :]

        samples_indices2 = np.random.randint(len(points), size=pert_number_points,)
        points2 = points[samples_indices2, :]
        normals2 = normals[samples_indices2, :]

        p_points = ([points1, points2]) #points3, points4]
        p_normals = ([normals1, normals2])# normals3, normals4]
        p_var = ([0.002, 0.00025])

        kdtree = cKDTree(points)
        pert_xyz, pert_normals, pert_gt_sdf = getSDF(kdtree, p_points, p_normals, p_var,  pert_number_points, rand_number_points, points, normals)

        xyz = pert_xyz
        normals_xyz = pert_normals
        gt_sdf = pert_gt_sdf 

        #convertToPLY(xyz, normals_xyz, gt_sdf, self.phase == 'val', self.filename)
        print("number points = ", len(xyz))

        return {'xyz': xyz, 'normals': normals_xyz, 'gt_sdf': gt_sdf}
    
    def __getitem__(self, index):
        data = self.data #np.load(self.getSDF(self.filename), allow_pickle=True).item()
        #cluster = self.cluster
        gt_sdf = data['gt_sdf']
        #percent = cluster['percent'] 

        posindex = torch.where(gt_sdf >= 0)[0]
        negindex = torch.where(gt_sdf < 0)[0]

        posxyz = data['xyz'][posindex]
        negxyz = data['xyz'][negindex]
        posnorm = data['normals'][posindex]
        negnorm = data['normals'][negindex]
        posgtsdf = gt_sdf[posindex]
        neggtsdf = gt_sdf[negindex]
        #print(self.args.subsample, flush=True)

        #self.subsample = 16384 #len(gt_sdf)
        #half = int(len(gt_sdf) / 2)
        half = int(self.args.subsample / 2)
        
        posindex = np.random.randint(len(posxyz), size=(half,))
        negindex = np.random.randint(len(negxyz), size=(self.args.subsample-half,))
        #convertToPLY(posxyz[posindex], fname='pos_test')
        #convertToPLY(negxyz[negindex], fname='neg_test')
        #exit()
        return {'index': index, 'data': torch.tensor(np.vstack((np.hstack((posxyz[posindex], posnorm[posindex], posgtsdf[posindex])), np.hstack((negxyz[negindex], negnorm[negindex], neggtsdf[negindex])))))}
        #return {'index': index, 'data': torch.tensor(np.vstack((np.hstack((posxyz[posindex], posnorm[posindex], posgtsdf[posindex], percent[posindex])), np.hstack((negxyz[negindex], negnorm[negindex], neggtsdf[negindex], percent[negindex])))))}


class DeepSdfDataset(data.Dataset):
    def __init__(self, fnames, args, phase='train'):
        self.phase = phase
        self.filenames = fnames
        self.numfiles = len(fnames)
        self.subsample = args.subsample
        self.datadir = args.datadir
        self.device = 'cuda'
        self.gtsdfdir = args.gtsdfdir
        self.initialize()

    def __len__(self):
        return self.numfiles

    def initialize(self):
        for fname in self.filenames:
            fname = fname[:-4]
            #print(fname)
            gtsdffile = os.path.join(self.gtsdfdir, fname+'.npy')
            if not os.path.exists(gtsdffile):
                gtsdfdata = self.getSDF(fname)
                np.save(gtsdffile, gtsdfdata)

    def getSDF(self, fname):
        gtsdffile = os.path.join(self.gtsdfdir, fname+'.npy')
        if os.path.exists(gtsdffile):
            return gtsdffile

        npyfile = np.load(os.path.join(self.datadir, fname+'.npy'), allow_pickle=True)

        points = torch.Tensor(npyfile[:,0:3])
        normals = torch.Tensor(npyfile[:,3:])
        number_points = points.shape[0]
              
        pert_number_points = int(len(points)/2)
        rand_number_points = int(0.1* len(points))

        samples_indices1 = np.random.randint(len(points), size=pert_number_points,)
        points1 = points[samples_indices1, :]
        normals1 = normals[samples_indices1, :]

        samples_indices2 = np.random.randint(len(points), size=pert_number_points,)
        points2 = points[samples_indices2, :]
        normals2 = normals[samples_indices2, :]

        p_points = [points1, points2] #points3, points4]
        p_normals = [normals1, normals2]# normals3, normals4]
        p_var = [0.002, 0.00025]

        kdtree = cKDTree(points)
        pert_xyz, pert_normals, pert_gt_sdf = getSDF(kdtree, p_points, p_normals, p_var,  pert_number_points, rand_number_points, points, normals)

        xyz = pert_xyz
        normals_xyz = pert_normals
        gt_sdf = pert_gt_sdf 

        return {'xyz': xyz, 'normals': normals_xyz, 'gt_sdf': gt_sdf}
    
    def __getitem__(self, index):
        data = np.load(self.getSDF(self.filenames[index][:-4]), allow_pickle=True).item()
        gt_sdf = data['gt_sdf']

        posindex = torch.where(gt_sdf >= 0)[0]
        negindex = torch.where(gt_sdf < 0)[0]

        posxyz = data['xyz'][posindex]
        negxyz = data['xyz'][negindex]
        posnorm = data['normals'][posindex]
        negnorm = data['normals'][negindex]
        posgtsdf = gt_sdf[posindex]
        neggtsdf = gt_sdf[negindex]
        #half = int(len(gt_sdf) / 2)
        half = int(self.subsample / 2)
        
        posindex = np.random.randint(len(posxyz), size=(half,))
        negindex = np.random.randint(len(negxyz), size=(self.subsample-half,))
        convertToPLY(posxyz[posindex], fname='pos_test')
        convertToPLY(negxyz[negindex], fname='neg_test')
        return {'index': index, 'data': torch.tensor(np.vstack((np.hstack((posxyz[posindex], posnorm[posindex], posgtsdf[posindex])), np.hstack((negxyz[negindex], negnorm[negindex], neggtsdf[negindex])))))}

class gridData(data.Dataset):
    def __init__(self, grid_N=128, BB = 1): #args=None):
        self.max_dimensions = np.ones((3, )) * float(BB)
        self.min_dimensions = -np.ones((3, )) * float(BB)

        seed()

        bounding_box_dimensions = self.max_dimensions - self.min_dimensions  # compute the bounding box dimensions of the point cloud
        #print("bounding box dim = ",bounding_box_dimensions)
        #grid_spacing = max(bounding_box_dimensions) / (args.grid_N - 9)  # each cell in the grid will have the same size
        #dim = np.arange(self.min_dimensions[0] - grid_spacing*4, self.max_dimensions[0] + grid_spacing*4, grid_spacing)
        grid_spacing = max(bounding_box_dimensions) / grid_N  # each cell in the grid will have the same size
        dim = np.arange(self.min_dimensions[0] - grid_spacing, self.max_dimensions[0] + grid_spacing, grid_spacing)
        X, Y, Z = np.meshgrid(list(dim), list(dim), list(dim))

        self.grid_shape = X.shape
        self.batchsize = grid_N
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

