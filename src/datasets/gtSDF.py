import torch
from numpy.random import normal as npnormal, uniform as npuniform
from numpy import sqrt as npsqrt, clip as npclip
from src.utils.utils import convertToPLY



def getSignedsdf(points, surf_points, surf_normals, nearest_index, sample_variance):
    gt_sdf = []
    sign = []
    for i in range(11):
        ray_vec = points - surf_points[nearest_index[:,i]]
        ray_vec_len = ray_vec.norm(dim=-1)
        index_normal = surf_normals[nearest_index[:,i]]
        if i == 0:
            gt_sdf = ray_vec_len
            index = torch.where(ray_vec_len < sample_variance)[0]
            dot = torch.bmm(index_normal[index].unsqueeze(dim=1), ray_vec[index].unsqueeze(dim=-1)).squeeze()
            gt_sdf[index] = torch.abs(dot) # torch.abs(index_normal[index].dot(ray_vec[index]))
        ray_vec = torch.nn.functional.normalize(ray_vec, dim=-1) 
        sign.append(-torch.sign(torch.bmm(index_normal.unsqueeze(dim=1), (ray_vec.unsqueeze(dim=-1))).squeeze()))

    sign = torch.stack(sign).transpose(-1,0)
    pos_sign = torch.count_nonzero(sign>0, dim=-1) 
    neg_sign = torch.count_nonzero(sign<0, dim=-1) 
    sign = (pos_sign > neg_sign).int()
    sign[torch.where(sign == 0)] = -1
    gt_sdf =(sign*gt_sdf)
    return gt_sdf


def getSDF(kdtree, p_points, p_normals, p_var,  number_points, number_rand_points, orig_points, orig_normals, BB=1):
    pert_var = []
    for p,n,v in zip(p_points, p_normals, p_var):
        clip = 0.05 if v == 0.002 else 0.01
        variance = torch.tensor(npclip(npnormal(0, npsqrt(v), size=(number_points, 3)), -clip, clip))
        #variance = npclip(npnormal(0, npsqrt(v),size=(number_points, 1)), -clip, clip)
        #pert_var.append(p + n*variance)
        pert_var.append(p + variance)

    pert_var = torch.vstack(pert_var)

    #print("number of pert =",len(pert_var))

    rand_points = npuniform(-BB, BB,(number_rand_points,3))
    rand_points = torch.tensor(rand_points)

    pert_normals = [] #torch.cat((normals, normals, rand_normals))
    #print("nearest_index = ", nearest_index)
    pointsToRemove = []
    rand_gt_sdf, nearest_index = kdtree.query(rand_points,k=1)
    ray_vec = rand_points - orig_points[nearest_index]
    ray_vec_len = ray_vec.norm(dim=-1)

    pert_points = torch.cat((pert_var, rand_points))
    gt_sdf, nearest_index = kdtree.query(pert_points,k=11)
    sign = []
    for i in range(11):
        ray_vec = pert_points - orig_points[nearest_index[:,i]]
        ray_vec_len = ray_vec.norm(dim=-1)
        #index_normal = torch.tensor(orig_normals[nearest_index[:,i]])
        index_normal = torch.Tensor(orig_normals[nearest_index[:,i]]).float()
        if i == 0:
            gt_sdf = ray_vec_len
        ray_vec = torch.nn.functional.normalize(ray_vec, dim=-1).float() 
        if i == 0:
            pert_normals.append(ray_vec)
        sign.append(torch.sign(torch.bmm(index_normal.unsqueeze(dim=1), (ray_vec.unsqueeze(dim=-1))).squeeze()))

    sign = torch.stack(sign).transpose(-1,0)
    #print(len(sign))
    pos_sign = torch.count_nonzero(sign>0, dim=-1) 
    neg_sign = torch.count_nonzero(sign<0, dim=-1) 
    
    sign = (pos_sign > neg_sign).int()
    sign[torch.where(sign == 0)[0]] = -1
    #print(sign)
    #print(len(sign))
    gt_sdf =(sign*gt_sdf)
    #print("hi", flush=True)
    #print(len(gt_sdf))
    
    possign = torch.where(gt_sdf > 0)[0]
    negsign = torch.where(gt_sdf < 0)[0]
    #print(possign)
    #print(negsign)
    #print(len(possign))
    #print(len(negsign))

    pert_normals = torch.vstack(pert_normals)
    #print(pert_points)
    #print(pert_normals)
    
    #convertToPLY(pert_points.squeeze()[possign], pert_normals.squeeze()[possign], isVal=False, fname='pos')
    #convertToPLY(pert_points.squeeze()[negsign], pert_normals.squeeze()[negsign], isVal=False, fname='neg')
    #exit()
    return pert_points, pert_normals, gt_sdf.unsqueeze(dim=-1)


