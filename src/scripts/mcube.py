import numpy as np
from skimage import measure
import torch
import torch.backends.cudnn as cudnn
from utils import normalize_pts_withdia

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)
# testing function

def getImplicitCurvatureforSamples(model, latent, samples, args):
    gaussCurvature = np.zeros((len(samples), ))
    model.eval()
    start_idx = 0
    numbatch = int(np.ceil(len(samples)/args.train_batch))
    print("numbatch sdf and curvature = ", numbatch)
    for i in range(numbatch):
        end = np.min((len(samples), (i+1)*args.train_batch))
        #print("{} {}".format(i*args.train_batch, end))
        xyz_tensor = torch.tensor(samples[i*args.train_batch:end]).to(device)
        this_bs = xyz_tensor.shape[0]
        xyz_tensor.requires_grad = True
        end_idx = start_idx + this_bs
        #with torch.no_grad():
        #latvec = latent(torch.tensor(0)).expand(this_bs,-1).to(device)
        latvec = latent.expand(this_bs,-1).to(device)
        pred_sdf_tensor = model(torch.cat([latvec,xyz_tensor.float()],dim=-1))

        predicted_gradient, hessian_matrix = getGradientAndHessian(pred_sdf_tensor, xyz_tensor)
        #sigma = (torch.linalg.svd(hessian_matrix)[1]).sum(dim=1)
        gauss = gaussianCurvature(predicted_gradient, hessian_matrix)

        gaussCurvature[start_idx:end_idx] = gauss.data.cpu().numpy()
        start_idx = end_idx

    return gaussCurvature

def getGaussCurvatureforSamples(model, latent, samples, args):
    gaussCurvature = np.zeros((len(samples), ))
    points = np.zeros((len(samples),3 ))

    start_idx = 0
    numbatch = int(np.ceil(len(points)/args.train_batch))
    print("numbatch sdf and curvature = ", numbatch)
    for i in range(numbatch):
        end = np.min((len(points), (i+1)*args.train_batch))
        #print("{} {}".format(i*args.train_batch, end))
        xyz_tensor = torch.tensor(samples[i*args.train_batch:end]).to(device)
        this_bs = xyz_tensor.shape[0]
        xyz_tensor.requires_grad = True
        end_idx = start_idx + this_bs
        latvec = latent.expand(this_bs,-1).to(device)
        pred_sdf_tensor = model(torch.cat([latvec,xyz_tensor.float()],dim=-1))

        predicted_gradient, hessian_matrix = getGradientAndHessian(pred_sdf_tensor, xyz_tensor)
        gauss = gaussianCurvature(predicted_gradient, hessian_matrix)

        points[start_idx:end_idx] = xyz_tensor.detach().cpu().numpy()
        gaussCurvature[start_idx:end_idx] = gauss.data.detach().cpu().numpy()
        start_idx = end_idx

    return points, gaussCurvature

def getSDFandCurvatureforSamples(model, latent, samples, args):
    IF = np.zeros((len(samples), ))
    SVD = np.zeros((len(samples), ))
    gaussCurvature = np.zeros((len(samples), ))
    meanCurv = np.zeros((len(samples), ))
    points = np.zeros((len(samples),3 ))

    start_idx = 0
    numbatch = int(np.ceil(len(IF)/args.train_batch))
    print("numbatch sdf and curvature = ", numbatch)
    for i in range(numbatch):
        end = np.min((len(IF), (i+1)*args.train_batch))
        #print("{} {}".format(i*args.train_batch, end))
        xyz_tensor = torch.tensor(samples[i*args.train_batch:end]).to(device)
        this_bs = xyz_tensor.shape[0]
        xyz_tensor.requires_grad = True
        end_idx = start_idx + this_bs
        latvec = latent.expand(this_bs,-1).to(device)
        pred_sdf_tensor = model(torch.cat([latvec,xyz_tensor.float()],dim=-1))

        predicted_gradient, hessian_matrix = getGradientAndHessian(pred_sdf_tensor, xyz_tensor)
        sigma = (torch.linalg.svd(hessian_matrix)[1]).sum(dim=1)
        gauss = gaussianCurvature(predicted_gradient, hessian_matrix)
        mean = meanCurvature(predicted_gradient, hessian_matrix)

        pred_sdf = pred_sdf_tensor.detach().cpu().squeeze().numpy()
        IF[start_idx:end_idx] = pred_sdf
        points[start_idx:end_idx] = xyz_tensor.detach().cpu().numpy()
        gaussCurvature[start_idx:end_idx] = gauss.data.detach().cpu().numpy()
        meanCurv[start_idx:end_idx] = mean.data.detach().cpu().numpy()
        SVD[start_idx:end_idx] = sigma.data.detach().cpu().numpy()
        start_idx = end_idx

    return points, IF, SVD, gaussCurvature, meanCurv

def getSDFforSamples(model, samples, args):
    IF = np.zeros((len(samples), ))
    start_idx = 0
    numbatch = int(np.ceil(len(IF)/args.train_batch))
    print("numbatch = ", numbatch)
    for i in range(numbatch):
        end = np.min((len(IF), (i+1)*args.train_batch))
        #print("{} {}".format(i*args.train_batch, end))
        xyz_tensor = torch.tensor(samples[i*args.train_batch:end]).to(device)
        this_bs = xyz_tensor.shape[0]
        xyz_tensor.requires_grad = True
        end_idx = start_idx + this_bs
        pred_sdf_tensor = model(xyz_tensor.float())
        pred_sdf = pred_sdf_tensor.detach().cpu().squeeze().numpy()
        IF[start_idx:end_idx] = pred_sdf
        start_idx = end_idx

    return IF

def getmcubePoints(dataset, model):
    model.eval()  # switch to test mode
    num_batch = len(dataset)
    #print("num batch", num_batch, flush=True)
    number_samples = dataset.number_samples
    grid_shape = dataset.grid_shape
    IF = np.zeros((number_samples, ))
    start_idx = 0
    for i in range(num_batch):
        data = dataset[i]  # a dict
        #xyz_tensor = data['xyz'].to(device)
        xyz_tensor = data.to(device)
        this_bs = xyz_tensor.shape[0]
        end_idx = start_idx + this_bs
        with torch.no_grad():
                pred_sdf_tensor = model(xyz_tensor)
        pred_sdf = pred_sdf_tensor.cpu().squeeze().numpy()
        IF[start_idx:end_idx] = pred_sdf
        start_idx = end_idx
    IF = np.reshape(IF, grid_shape)
    #print("min and max of IF = {} {}".format(IF.min(), IF.max()))
    #print("min and max of IF = {} {}".format(IF.min(), IF.max()))
    mid = (IF.max() + IF.min())/2
    print("mid = ",mid, flush=True)
    if((IF.min() >= 0 or IF.max() <= 0 or np.isnan(np.array(IF.min())) or np.isnan(IF.max()))):
        print("IF level not in 0 ")
        #verts, simplices, normals, values = measure.marching_cubes(IF,mid) #,0) #,0) #,mid)
        return [], []
    
    verts, simplices, normals, values = measure.marching_cubes(IF,0) #,0) #,0) #,mid)

    v = np.array(verts)
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('z', 90, degrees=True)
    rotv = r.apply(v)
    rotv[:,0] = -rotv[:,0]
    rotv = normalize_pts_withdia(rotv)
    #print(max(rotv))
    #print(min(rotv.any()))
    simplices = simplices[:,::-1]
    
    return rotv, simplices
