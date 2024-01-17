#import open3d
import torch
import torch.backends.cudnn as cudnn
from .loss import datafidelity_loss, implicit_loss
from src.tools.gradient import getGradient, getGradientAndHessian
from src.tools.curvature import gaussianCurvature, meanCurvature, getPrincipalCurvatures
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
    def __init__(self, device, args):
        super(Model, self).__init__()
        self.device = device
        self.net = MLPNet(channels=[512,512,512,512,512,512,512,512], latcode=0, arch=args.arch, dropout=args.dropout, lasttanh=args.lasttanh, dprob=args.dprob)
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
        model_path = None
        if self.args.reg:
            model_path = os.path.join(self.args.pretrained_folder, self.args.resume_checkpoint)
        elif self.args.resume:
            #model_path = os.path.join(self.args.checkpoint_folder, self.args.resume_checkpoint)
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

    def run(self, batch, computeGauss=False, isTrain=True):
        loss_sum = 0.0
        loss_count = 0.0
        indexcount = 0

        data = torch.Tensor(batch['data'].float()).to(self.device)
        #index = torch.tensor(batch['index']).to(self.device)

        sampled_points = data[:,:,0:3].squeeze().to(device) # data['xyz'].to(device)
        sampled_points.requires_grad = True
        this_bs =  sampled_points.shape[0]
        predicted_sdf = self.net(sampled_points)
        gt_sdf_tensor = torch.clamp(data[:,:,6].view(-1,1).to(device), -self.args.clamping_distance, self.args.clamping_distance)
        gt_normal = data[:,:,3:6].squeeze().to(device)
        loss = datafidelity_loss(predicted_sdf, gt_sdf_tensor, self.args)
        if self.args.reg :
            predicted_gradient, pred_hess_matrix = getGradientAndHessian(predicted_sdf, sampled_points)
            hessloss = implicit_loss(predicted_gradient, pred_hess_matrix, self.args, self.device)
            loss = loss + hessloss
        gauss = None
        if computeGauss:
            if self.args.reg == 0:
                predicted_gradient, pred_hess_matrix = getGradientAndHessian(predicted_sdf, sampled_points)
            with torch.no_grad():
                gauss = gaussianCurvature(predicted_gradient, pred_hess_matrix)
                gauss = torch.abs(gauss)
        return loss, gauss

    def runtest(self, batch, computeGauss=False, isTrain=False):
        indexcount = 0
        data = torch.Tensor(batch['data'].float()).to(self.device)
        sampled_points = data[:,:,0:3].squeeze().to(device) # data['xyz'].to(device)
        sampled_points.requires_grad = True
        this_bs =  sampled_points.shape[0]
        predicted_sdf = self.net(sampled_points)
        gt_sdf_tensor = torch.clamp(data[:,:,6].view(-1,1).to(device), -self.args.clamping_distance, self.args.clamping_distance)
        predicted_gradient, pred_hess_matrix = getGradientAndHessian(predicted_sdf, sampled_points)
        if computeGauss:
            with torch.no_grad():
                gauss = gaussianCurvature(predicted_gradient, pred_hess_matrix)
                meancurv = meanCurvature(predicted_gradient, pred_hess_matrix)
                princurv = getPrincipalCurvatures(meancurv, gauss)
        return torch.abs(gauss), torch.abs(meancurv), princurv

    def runtestondata(self, data, computeGauss=False, isTrain=False):
        indexcount = 0
        data = torch.Tensor(data).to(self.device)
        sampled_points = data.squeeze().to(device) # data['xyz'].to(device)
        sampled_points.requires_grad = True
        this_bs =  sampled_points.shape[0]
        predicted_sdf = self.net(sampled_points)
        gt_sdf_tensor = torch.clamp(torch.zeros(data.shape[0]).view(-1,1).to(device), -self.args.clamping_distance, self.args.clamping_distance)
        predicted_gradient, pred_hess_matrix = getGradientAndHessian(predicted_sdf, sampled_points)
        if computeGauss:
            with torch.no_grad():
                determinant, gauss = gaussianCurvature(predicted_gradient, pred_hess_matrix)
                meancurv = meanCurvature(predicted_gradient, pred_hess_matrix)
                princurv = getPrincipalCurvatures(meancurv, gauss)
        return torch.abs(gauss), torch.abs(meancurv), princurv, determinant
