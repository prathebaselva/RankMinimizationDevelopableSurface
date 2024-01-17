import math
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn



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


def datafidelity_loss(predicted_sdf, gt_sdf_tensor, latent_codes, args):
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

def implicit_loss(gradient, hessian_matrix, args, device):

    hess_regularizer = torch.tensor(0).to(device)
    sdfloss = torch.tensor(0).to(device)
    SVD = torch.tensor(0)
    n = gradient.shape[0]

    hatHMatrix = torch.zeros(n, 4, 4)
    hatHMatrix[:, 0:3, 0:3] = hessian_matrix
    hatHMatrix[:, 0:3, 3] = gradient
    hatHMatrix[:, 3:, 0:3] = gradient.view(-1,1,3)
    
    print(hatHMatrix[0])
    print(hessian_matrix[0])
    print(gradient[0])

    if args.hess_delta:
        hess_regularizer = torch.tensor(2e20).to(device)
        #U1,SVD1,V1 = customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
        U1,SVD1,V1 = (torch.linalg.svd(hatHMatrix))
        SVD = (SVD1).sum(dim=1)
        if args.losstype == 'svd':
            hess_regularizer = args.hess_delta * SVD.mean()
        if args.losstype == 'psum':
            hess_regularizer = args.hess_delta * SVD1[:,2:].mean()
        if args.losstype == 'svd3':
            hess_regularizer = args.hess_delta * SVD1[:,2:].mean()
        if args.losstype == 'detsvd3':
            hess_regularizer = args.hess_delta * SVD1[:,2:].mean()
        if args.losstype == 'invsvd':
            hess_regularizer = args.hess_delta * (1/(1e-10+SVD)).mean()

        if args.losstype == 'logdet':
            sign, detvalue = torch.linalg.slogdet(hessian_matrix)
            hess_regularizer = args.hess_delta * torch.abs(detvalue).mean()

        if args.losstype == 'hessiandet':
            L = torch.real(torch.linalg.eigvals(hatHMatrix))
            numnegatives = torch.sum(L < 0, dim=1)
            poseig = torch.where(numnegatives == 0)[0]
            print(len(poseig))

        if args.losstype == 'logdetT':
            #detvalue = torch.logdet(torch.bmm(hessian_matrix.permute(0,2,1), hessian_matrix) + torch.eye(3).to(device).repeat(n,1,1))
            #print(hessian_matrix, flush=True)
            #hessbmm = torch.bmm(hessian_matrix.permute(0,2,1), hessian_matrix)
            #print(hessbmm.shape, flush=True)
            L = torch.real(torch.linalg.eigvals(hatHMatrix))
            numzeros = torch.sum(L, dim=1)
            zeroeig = torch.where(numzeros == 0)[0]
            print(len(zeroeig))
            print(hatHMatrix[zeroeig])
            numnegatives = torch.sum(L < 0, dim=1)
            poseig = torch.where(numnegatives == 0)[0]
            print(len(poseig))
            print(hatHMatrix[poseig])
            negeig = torch.where(numnegatives > 0)[0]
            print(len(negeig))
            print(hatHMatrix[negeig])
            exit()

            #print(numnegatives, flush=True)
            #print(len(numnegatives), flush=True)
            negeig = torch.where(numnegatives > 0)[0]
            if len(negeig) > 0:
                print(len(negeig))
                print((negeig))
                print(L[negeig])
                print(SVD1[negeig])
                print(hessian_matrix[negeig])
                exit()
            
            hessbmm = hessbmm + (torch.eye(3).repeat(args.subsample,1,1)).to(device)
            detvalue = torch.logdet(hessbmm)
            nanindex = torch.where(torch.isnan(detvalue) == 1)[0]
            if len(nanindex) > 0:
                print(nanindex)
                print(hessbmm[nanindex], flush=True)
                exit()

            hess_regularizer = args.hess_delta * torch.abs(detvalue).mean() 
            if torch.isnan(hess_regularizer):
                print(detvalue[negdet])
                print(detvalue)
                print(SVD1)
                exit()
        if args.losstype == 'eikonal':
            hess_regularizer = args.hess_delta * torch.abs(gradient.norm(dim=-1)-1).mean()

#    eikonalloss = torch.tensor(0).to(device)
#    if args.imp_eikonal_delta:
        #eikonalloss = args.imp_eikonal_delta * torch.abs(gradient.norm(dim=-1) -1e-5).mean()
 #       eikonalloss = args.imp_eikonal_delta * torch.abs(gradient.norm(dim=-1) -1).mean()

#    if iteration:
#        print("hessloss =",hess_regularizer)
#        print("sdfloss = ", sdfloss)
#        print("eikonalloss = ", eikonalloss)
#
#    if torch.isnan(hess_regularizer):
#        print("valid indices = ",len(valid_indices))
#    if torch.isnan(eikonalloss):
#        print("eikonalloss ", gradient)
    #dataloss = 0 
    #dataloss = 0 
    #loss = hess_regularizer +  eikonalloss
    loss = hess_regularizer
    return loss 
