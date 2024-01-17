#import open3d
import torch
import torch.backends.cudnn as cudnn
from numpy import arange
import torch.nn as nn
import os

from network import MLPNet

outfolder = 'output/'



class Model(nn.Module):
    def __init__(self, model_path, device):
        super(Model, self).__init__()
        self.device = device
        self.net = MLPNet(channels=[512,512,512,512,512,512,512,512], latcode=0, arch='gelu', dropout=1, lasttanh=False)
        self.load_model(model_path)

    def model_dict(self):
        return {
            'net': self.net.state_dict()
        }

    def load_model(self, model_path):
        if model_path is not None:
            if os.path.exists(model_path):
                print("Training model found", flush=True)
                checkpoint = torch.load(model_path)
                if 'net' in checkpoint:
                    print('net')
                    self.net.load_state_dict(checkpoint['net'])

    def run(self, sampled_points):
        predicted_sdf = self.net(sampled_points)
        return predicted_sdf
