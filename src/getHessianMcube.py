import torch
import math
import numpy as np
from gradient import getGradientAndHessian
deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)

def getHessianforwnnm(model, sampled_points):
    start_idx = 0
    bs = 512
    numbatch = int(math.ceil(len(sampled_points)/bs))
    hesswnnm = torch.zeros((len(sampled_points),3,3))
    print("numbatch= ", numbatch)
    for i in range(numbatch):
        end = np.min((len(sampled_points), (i+1)*bs))
        xyz_tensor = torch.tensor(sampled_points[i*bs:end]).to(device)
        this_bs = xyz_tensor.shape[0]
        xyz_tensor.requires_grad = True
        pred_sdf_tensor = model(xyz_tensor.float())
        end_idx = start_idx + this_bs
        #pred_sdf_tensor = torch.tensor(predicted_sdf[i*bs:end]).to(device)

        predicted_gradient, hessian_matrix = getGradientAndHessian(pred_sdf_tensor, xyz_tensor)
        hesswnnm[start_idx:end_idx] = hessian_matrix.data
        start_idx = end_idx
        #gradwnnm[start_idx:end_idx] = predicted_gradient.data

    hesswnnm = hesswnnm.to(device)
    return  hesswnnm
    

