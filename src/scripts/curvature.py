import numpy as np
import torch

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)

def getPrincipalCurvatures(H,K):
    s = (H)**2 - (K)
    pos = torch.where(s > 0)[0]
    k1 = torch.abs(H[pos] + s[pos])
    k2 = torch.abs(H[pos] - s[pos])
    return torch.max(k1,k2), torch.min(k1,k2)

def meanCurvature(jacobian_matrix, hessian_matrix):
    j = jacobian_matrix.unsqueeze(dim=1)
    j_norm = jacobian_matrix.norm(dim=-1)
    num = torch.zeros(len(jacobian_matrix),4,4).to(device)
    num[:,:3,:3] = hessian_matrix
    num[:,3:,:3] = j
    num[:,:3,3:4] = j.reshape(len(j),3,1)

    H = (num.diagonal(offset=0,dim1=-1, dim2=-2).sum(-1))/2
#    num1 = torch.mul(torch.matmul(j, hessian_matrix).squeeze(), jacobian_matrix).sum(dim=-1)
#    trace = torch.diagonal(hessian_matrix, dim1=-2, dim2=-1).sum(-1)
#    den = j_norm**3
#    H = (num1 - (j_norm**2)*trace)/(2*den)
    return torch.abs(H)
    #return H

def gaussianCurvature(jacobian_matrix, hessian_matrix):
    #print(jacobian_matrix.shape)
    #print(hessian_matrix.shape)
    j = jacobian_matrix.unsqueeze(dim=1)
    j_norm = jacobian_matrix.norm(dim=-1)
    num = torch.zeros(len(jacobian_matrix),4,4).to(device)
    #print(torch.count_nonzero(hessian_matrix))
    num[:,:3,:3] = hessian_matrix
    num[:,3:,:3] = j
    num[:,:3,3:4] = j.reshape(len(j),3,1)

    det_num = torch.linalg.det(num)
    det_den = j_norm**4
    K = (-(det_num/(1e-10+det_den)))
    #print(K)
    
    return det_num, torch.abs(K)
    #return K

