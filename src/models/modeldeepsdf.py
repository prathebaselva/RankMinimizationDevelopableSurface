#import open3d
import torch
import torch.backends.cudnn as cudnn
from .loss import datafidelity_loss, implicit_loss
from src.tools.gradient import getGradient, getGradientAndHessian
from src.tools.curvature import gaussianCurvature
from torch.autograd import Variable
from numpy import arange
from loguru import logger
import random
import torch.nn as nn
import os

from .network import MLPNet

outfolder = 'output/'

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)


class Model(nn.Module):
    def __init__(self, latent, device, args):
        super(Model, self).__init__()
        self.device = device
        self.latent = latent
        self.latcodesize = args.latcode
        self.net = MLPNet(channels=[512,512,512,512,512,512,512,512], latcode=args.latcode, arch=args.arch, dropout=args.dropout, lasttanh=args.lasttanh)
        self.args = args
        self.reg = self.args.reg
        self.load_model()

    def set_val_latent(self, val_latent):
        self.val_latent = val_latent
        self.val_latent = self.val_latent.to(self.device)

    def set_test_latent(self, test_latent):
        self.test_latent = test_latent.to(self.device)

    def model_dict(self):
        return {
            'net': self.net.state_dict()
        }

    def load_model(self):
        print("in load model", flush=True)
        model_path = None
        #if self.args.reg:
        #    model_path = os.path.join(self.args.pretrained_folder, self.args.resume_checkpoint)
        #elif self.args.resume:
        model_path = os.path.join(self.args.pretrained_folder, self.args.resume_checkpoint)

        if model_path is not None:
            if os.path.exists(model_path):
                logger.info('Training model {model_path} found')
                checkpoint = torch.load(model_path)
                if 'net' in checkpoint:
                    print('net')
                    self.net.load_state_dict(checkpoint['net'])
            else:
                logger.info('no {model_path} found')

    def run(self, batch, computeGauss=False):
        loss_sum = 0.0
        loss_count = 0.0
        indexcount = 0

        data = torch.Tensor(batch['data'].float()).to(self.device)
        index = torch.tensor(batch['index']).to(self.device)

        sampled_points = data[:,:,0:3].to(device) # data['xyz'].to(device)
        sampled_points.requires_grad = True
        this_bs =  sampled_points.shape[0]

        self.latent.requires_grad = True
        if self.training:
            lat_vec = self.latent(index).to(self.device)
        else:
            self.test_latent.requires_grad = True
            #print("testing", flush=True)
            lat_vec = self.test_latent.to(self.device)
            #print(lat_vec)

        lat_vec = lat_vec.repeat(1,data.shape[1]).reshape(-1,data.shape[1],lat_vec.shape[-1])
        sampledlat_points = torch.cat([lat_vec, sampled_points],dim=-1).view(-1, self.latcodesize+3).to(self.device)
        if self.training:
            predicted_sdf = self.net(sampledlat_points)
        else:
            #with torch.no_grad():
            predicted_sdf = self.net(sampledlat_points)
        gt_sdf_tensor = torch.clamp(data[:,:,6].view(-1,1).to(device), -self.args.clamping_distance, self.args.clamping_distance)
        loss = datafidelity_loss(predicted_sdf, gt_sdf_tensor, self.args)
        #print(loss, flush=True)
        if self.args.reg :
            #index = torch.where(torch.abs(predicted_sdf) <= 0.018)[0]
            #print(len(index))
            #index = torch.where(predicted_sdf < 1e-3)[0]
            #print(len(index))
            #index = torch.where(predicted_sdf < 1e-4)[0]
            #print(len(index))
            #print(predicted_sdf.shape)
            #print(sampled_points.shape)
            #predicted_gradient, pred_hess_matrix = getGradientAndHessian(predicted_sdf, sampled_points.squeeze())
            predicted_gradient, pred_hess_matrix = getGradientAndHessian(predicted_sdf, sampledlat_points)
            #print(pred_hess_matrix[0])
            #print(pred_hess_matrix.shape)
            #print(predicted_gradient.shape, flush=True)
            hessloss = implicit_loss(predicted_gradient.squeeze()[:,256:], pred_hess_matrix.squeeze()[:,:,256:], self.args, self.device)
            #print(hessloss.requires_grad)
            #hessloss.retain_grad()
            #hessloss.backward(retain_graph=True)
            #print("ok")
            #exit()
            #hessloss.grad = customgrad()
            loss = loss + hessloss
        gauss = None
        if computeGauss:
            if self.args.reg == 0:
                predicted_gradient, pred_hess_matrix = getGradientAndHessian(predicted_sdf, sampled_points)
            with torch.no_grad():
                print("getting gauss")
                gauss = gaussianCurvature(predicted_gradient, pred_hess_matrix)
                #allgauss.append(gauss)
                #gauss = torch.stack(allgauss)
        return loss, gauss
