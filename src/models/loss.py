import math
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from .customgrad import SingularValueGrad



#def datafidelity_lossnormal(predicted_sdf, predicted_gradient, gt_sdf_tensor, surface_normals, surfaceP_indices, args):
def datafidelity_lossnormal(predicted_sdf, predicted_gradient, gt_sdf_tensor, surface_normals, args):

    pred_sdf = predicted_sdf.clone()
    numpos = len(torch.where(pred_sdf > 0)[0])
    numneg = len(torch.where(pred_sdf < 0)[0])
    dataloss = args.data_delta * torch.nn.functional.l1_loss(predicted_sdf, gt_sdf_tensor) 
    normal_reg_loss = args.norm_delta * ( 1- torch.nn.functional.cosine_similarity(predicted_gradient, surface_normals, dim=-1)).mean()
    #print(dataloss)
    #print(normal_reg_loss, flush=True)
    #normal_reg_loss = args.normal_delta * ( 1- torch.nn.functional.cosine_similarity(predicted_gradient[surfaceP_indices], surface_normals[surfaceP_indices], dim=-1)).mean()

    loss = dataloss + normal_reg_loss
    return loss

def datafidelity_loss(predicted_sdf, gt_sdf_tensor, args):
    dataloss = args.data_delta * torch.nn.functional.l1_loss(predicted_sdf, gt_sdf_tensor , reduction='mean')
    loss = dataloss
    return loss

def datafidelity_latentloss(predicted_sdf, gt_sdf_tensor, latent_codes, args):
    dataloss = args.data_delta * torch.nn.functional.l1_loss(predicted_sdf, gt_sdf_tensor , reduction='mean')
    codeloss = args.lat_delta * (torch.norm(latent_codes, dim=1)).mean()
    #print(dataloss)
    #print(codeloss, flush=True)
    loss = dataloss + codeloss

    return loss

def datafidelity_testloss(predicted_sdf, gt_sdf_tensor, latent_codes, epoch, args, gradient=None):
    dataloss = args.data_delta * torch.nn.functional.l1_loss(predicted_sdf, gt_sdf_tensor , reduction=args.data_reduction)
    #codeloss = args.code_delta* (torch.norm(latent_codes, dim=1)).mean()
    codeloss =  args.code_delta* torch.norm(latent_codes)
    if args.losstype == 'dataeikonal':
        eikonal_loss = args.eikonal_delta * ((gradient.norm(dim=-1)-1)**2).mean()
        loss = dataloss + codeloss + eikonal_loss
    else:
        loss = dataloss + codeloss

    return loss

def implicit_loss(gradient, hessian_matrix, args, device, index=None):
    hess_regularizer = torch.tensor(0).to(device)
    sdfloss = torch.tensor(0).to(device)
    SVD = torch.tensor(0)
    n = gradient.shape[0]

    hatHMatrix = torch.zeros(n, 4, 4)
    hatHMatrix[:, 0:3, 0:3] = hessian_matrix
    hatHMatrix[:, 0:3, 3] = gradient
    hatHMatrix[:, 3:, 0:3] = gradient.view(-1,1,3)
    hatHMatrix = hatHMatrix.to(device)

    if args.hess_delta:
        hess_regularizer = torch.tensor(2e20).to(device)
        #U,SVD,V = torch.linalg.svd(hessian_matrix) #customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
        #print(SVD[0:5], flush=True)
        if args.losstype == 'svdcus':
            sgrad = SingularValueGrad.apply
            hess_regularizer = args.hess_delta * sgrad(hessian_matrix).mean()
            print(hess_regularizer, flush=True)
            return hess_regularizer
        if args.losstype == 'svdcus1':
            sgrad = SingularValueGrad.apply
            hess_regularizer = args.hess_delta * sgrad(1, hessian_matrix).mean()
            print(hess_regularizer, flush=True)
            return hess_regularizer
        if args.losstype == 'svd':
            U,SVD,V = torch.linalg.svd(hessian_matrix) #customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
            hess_regularizer = args.hess_delta * torch.sum(SVD, dim=1).mean()
            print(hess_regularizer, flush=True)
            return hess_regularizer.mean()
        if args.losstype == 'svdhat':
            U,SVD,V = torch.linalg.svd(hatHMatrix) #customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
            hess_regularizer = args.hess_delta * torch.sum(SVD, dim=1).mean()
            print(hess_regularizer, flush=True)
            return hess_regularizer.mean()
        if args.losstype == 'svd3':
            U,SVD,V = torch.linalg.svd(hessian_matrix) #customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
            #print(SVD.shape)
            svdindex = torch.where(SVD[:,2]  > 0)[0]
            #print(SVD)
            #print(SVD[index])
            print(len(svdindex))
            #print(index)
            #print("------")
            #hess_regularizer = args.hess_delta * SVD[index][:,2:].mean()
            hess_regularizer = args.hess_delta * SVD[:,2:].mean()
            #hess_regularizer = args.hess_delta * SVD[:,2:].mean()
            print(hess_regularizer, flush=True)
        if args.losstype == 'svd2':
            U,SVD,V = torch.linalg.svd(hessian_matrix) #customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
            hess_regularizer = args.hess_delta * SVD[:,1:].mean()
            print(hess_regularizer, flush=True)
        if args.losstype == 'svd3hat':
            _,SVDhat,_ = (torch.linalg.svd(hatHMatrix))
            hess_regularizer = args.hess_delta * SVDhat[:,2:].mean()
            print(hess_regularizer, flush=True)
        if args.losstype == 'hessiandet':
            alldetval = (torch.abs(torch.linalg.det(hessian_matrix)))
            index = torch.where(alldetval != 0)[0]
            #print(len(index))
            hess_regularizer = args.hess_delta * alldetval[index].mean()
            #print(hess_regularizer, flush=True)
        if args.losstype == 'hessiandethat':
            #matrank = (torch.linalg.matrix_rank(hatHMatrix))
            #U,SVD,V = torch.linalg.svd(hatHMatrix) #customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
            #print(torch.max(SVD))
            #print(SVD)
            #print(matrank)
            alldetval = (torch.abs(torch.linalg.det(hatHMatrix)))
            index = torch.where(alldetval != 0)[0]
            #print(len(index))
            hess_regularizer = args.hess_delta * alldetval[index].mean()
            #print(hess_regularizer, flush=True)
        if args.losstype == 'hessiansvd3':
            alldetval = (torch.abs(torch.linalg.det(hatHMatrix)))
            index = torch.where(alldetval != 0)[0]
            #print(len(index))
            hess_regularizer = 3 * alldetval[index].mean()

            U,SVD,V = torch.linalg.svd(hessian_matrix) #customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
            hess_regularizer += 1e1 * SVD[:,2:].mean()
        if args.losstype == 'logdet':
            #alldetval = (torch.abs(torch.linalg.det(hessian_matrix)))
            #index = torch.where(alldetval != 0)[0]
            detval = (1e-7+torch.abs(torch.det(torch.bmm(hessian_matrix.permute(0,2,1), hessian_matrix) + torch.eye(3).to(device).repeat(n,1,1)))).log()
            #index = torch.where(detval != float('nan'))[0]
            #detval[detval == float('nan')] = 0
            #hth = torch.bmm(hessian_matrix.permute(0,2,1), hessian_matrix) + torch.eye(3).to(device).repeat(n,1,1)
            #detval = 2 * torch.linalg.cholesky(hth).diagonal(dim1=-2, dim2=-1).log().sum(-1)
            hess_regularizer = args.hess_delta * detval[index].mean()
            print(hess_regularizer, flush=True)

        if args.losstype == 'logdethat':
            #alldetval = (torch.abs(torch.linalg.det(hatHMatrix)))
            #index = torch.where(alldetval != 0)[0]
            hth = torch.bmm(hatHMatrix.permute(0,2,1), hatHMatrix) + torch.eye(4).to(device).repeat(n,1,1)
            detval = 2 * torch.linalg.cholesky(hth).diagonal(dim1=-2, dim2=-1).log().sum(-1)
            #detvalue = torch.logdet(torch.bmm(hatHMatrix[index].permute(0,2,1), hatHMatrix[index]) + torch.eye(4).to(device).repeat(n,1,1))
            hess_regularizer = args.hess_delta * detval.mean()
            print(hess_regularizer, flush=True)

        if args.losstype == 'eikonal':
            hess_regularizer = args.hess_delta * torch.abs(gradient.norm(dim=-1)-1).mean()
            print(hess_regularizer, flush=True)

    loss = hess_regularizer
    return loss 
