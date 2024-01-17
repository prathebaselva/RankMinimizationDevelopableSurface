#import open3d
import torch.backends.cudnn as cudnn
from loss import *
from curvature import *
from gradient import *
from trainhelper import *
from loadmodel import *
from utils import *
from dataset import *
#from gradient1 import *
#from hessian1 import *
from torch.autograd import Variable
from getHessianMcube import *

#outfolder = '/mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/output/'
outfolder = 'output/'

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)
   

def deepsdfrunmodel(points, gt_sdf, lat_vecs, indices, model, optimizer, epoch, args, mcube_points=None, mcube_sdf=None, isTrain=True):
    loss_sum = 0.0
    loss_count = 0.0
    regloss_sum = 0.0
    sdfloss_sum = 0.0
    data_pos_sum = 0
    data_neg_sum = 0
    mcube_pos_sum = 0
    mcube_neg_sum = 0
    mcube_zero_sum = 0
    num_batch = math.ceil(len(points)/512)
    indexcount = 0
    #print("num_batch=",num_batch)
    gaussCurvature = 0
    surf_gaussCurvature = 0
    gaussCurvature_sum = 0
    surf_gaussCurvature_sum = []
    points  = []
    IF = []
    SVD_all = []
    gaussCurvature_all = []
    meanCurvature_all = []
    
    surfaceP_points = []
    surfaceP_points_gradients = []
    surfaceP_points_svd = []

    interval = (epoch % 3 ==0)
    avgsdf = 0
    maxsdf = -1
    minsdf = 1e10
    maxsvd = 0
    minsvd = 0
    pinterval = np.random.randint(0,num_batch-1)

    batch_split = 64
    point_chunk = torch.chunk(points, batch_split)
    indices_chunk = torch.chunk(indices.unsqueeze(-1).repeat(1, 16384).view(-1),batch_split)
    gt_sdf_chunk = torch.chunk(gt_sdf, batch_split)

    for i in range(batch_split):
        optimizer.zero_grad()
        batch_vecs = lat_vecs(indices[i])
        sampled_points = torch.cat([batch_vecs, point_chunk[i]], dim=1)
        sampled_points.requires_grad = True

        this_bs =  sampled_points.shape[0]
        k = interval and (i == pinterval)
        predicted_sdf = model(sampled_points)

        gt_sdf_tensor = torch.clamp(gt_sdf_chunk[i].to(device), -args.clamping_distance, args.clamping_distance)

        predicted_gradient = getGradient(predicted_sdf, sampled_points)
        loss = args.data_delta * torch.nn.functional.l1_loss(predicted_sdf, gt_sdf_tensor , reduction=args.data_reduction)

        if isTrain:
            #with torch.autograd.detect_anomaly():
            (loss).backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if printinterval:
                total_norm = 0
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        print("name = ", name)
                        param_norm = p.grad.data.norm(2)
                        print(param_norm)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm **(1./2)
                print("total norm = ", total_norm)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip, norm_type=2)
      
        loss_sum += loss.item() * this_bs
            
        loss_count += this_bs
        if isTrain:
            optimizer.step()

    if loss_count == 0:
        return 2e20
    return loss_sum, loss_count 



