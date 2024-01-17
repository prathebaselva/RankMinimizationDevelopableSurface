import os
import torch
import errno
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import directed_hausdorff as Hausdorff
matplotlib.use("Agg")



# function to save a checkpoint during training, including the best model so far
def save_curr_checkpoint(state, checkpoint_folder='checkpoints/', modelname='model'):
    checkpoint_file = os.path.join(checkpoint_folder, 'model_last_'+modelname+'.pth.tar')
    torch.save(state, checkpoint_file)

#def save_checkpoint(state, is_best_gauss, is_best_loss, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
def save_reg0checkpoint(state,  is_best_loss, checkpoint_folder='checkpoints/', modelname='model'):
    if is_best_loss:
        checkpoint_file = os.path.join(checkpoint_folder, 'model_best_'+modelname+'.pth.tar') 
        torch.save(state, checkpoint_file)

def save_checkpoint(state, is_best_discmean, is_best_discmedian, is_best_impmean, is_best_impmedian, is_best_loss, checkpoint_folder='checkpoints/',  modelname='model_best'):
    if is_best_discmean:
        checkpoint_file = os.path.join(checkpoint_folder, modelname+'discmean.pth.tar') 
        torch.save(state, checkpoint_file)
    if is_best_discmedian:
        checkpoint_file = os.path.join(checkpoint_folder, modelname+'discmedian.pth.tar') 
        torch.save(state, checkpoint_file)
    if is_best_impmean:
        checkpoint_file = os.path.join(checkpoint_folder, modelname+'impmean.pth.tar') 
        torch.save(state, checkpoint_file)
    if is_best_impmedian:
        checkpoint_file = os.path.join(checkpoint_folder, modelname+'impmedian.pth.tar') 
        torch.save(state, checkpoint_file)
    if is_best_loss:
        checkpoint_file = os.path.join(checkpoint_folder, modelname+'loss.pth.tar') 
        torch.save(state, checkpoint_file)
        #shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def isdir(dirname):
    return os.path.isdir(dirname)


def normalize_pts_withdia(input_pts):
    maxdim = np.max(input_pts, axis=0)
    mindim = np.min(input_pts, axis=0)
    diff = maxdim - mindim
    center = (maxdim+mindim)/2
    centered_pts = input_pts - center
    radius = np.max(diff)/2
    normalized_pts = centered_pts / radius
    
    #maxdim = np.max(normalized_pts, axis=0)
    #mindim = np.min(normalized_pts, axis=0)
    #print(" {} {}".format(maxdim, mindim))
    return  normalized_pts

def find_center_and_radius(input_pts):
    center_point = np.mean(input_pts, axis=0)
    center_point = center_point[np.newaxis, :]
    #print("center_point = ",center_point)
    centered_pts = input_pts - center_point

    largest_radius = np.amax(np.sqrt(np.sum(centered_pts ** 2, axis=1)))
    #print("largest_radius = ",largest_radius)
    return centered_pts, largest_radius


def normalize_pts(input_pts):
    centered_pts, largest_radius = find_center_and_radius(input_pts)
    normalized_pts = centered_pts / largest_radius   # / 1.03  if we follow DeepSDF completely

    return normalized_pts


def normalize_normals(input_normals):
    normals_magnitude = np.sqrt(np.sum(input_normals ** 2, axis=1))
    normals_magnitude = normals_magnitude[:, np.newaxis]

    normalized_normals = input_normals / (1e-5+normals_magnitude)

    return normalized_normals


def convertToPLY(points, normals=None, gt=None, isVal=False, fname=None):
    if fname is not None:
        if isVal:
            filename = fname+"_val.ply"
        else:
            filename = fname+"_train.ply"
    else:
        filename = "val.ply" if isVal else "train.ply" 
    plyfile = open(filename, "w")

    plyfile.write("ply\n")
    plyfile.write("format ascii 1.0\n")
    plyfile.write("element vertex "+str(len(points))+"\n")
    plyfile.write("property float x\n")
    plyfile.write("property float y\n")
    plyfile.write("property float z\n")
    if not normals is None:
        plyfile.write("property float nx\n")
        plyfile.write("property float ny\n")
        plyfile.write("property float nz\n")
    if not gt is None:
        plyfile.write("property float gt\n")
    plyfile.write("element face 0\n")
    plyfile.write("property list uchar int vertex_indices\n")
    plyfile.write("end_header\n")

    points = points.numpy()
    if (not normals is None) and (not gt is None):
        normals = normals.numpy()
        for p,n,g in zip(points,normals,gt):
            #plyfile.write(str(p[0])+" "+str(p[1])+" "+str(p[2])+" "+str(n[0])+" "+str(n[1])+" "+str(n[2])+"\n")
            plyfile.write(str(p[0])+" "+str(p[1])+" "+str(p[2])+" "+str(n[0])+" "+str(n[1])+" "+str(n[2])+" "+str(g.item())+"\n")
    elif not normals is None:
        for p,n in zip(points,normals):
            plyfile.write(str(p[0])+" "+str(p[1])+" "+str(p[2])+" "+str(n[0].item())+" "+str(n[1].item())+" "+str(n[2].item())+"\n")
    else:
        for p in points:
            plyfile.write(str(p[0])+" "+str(p[1])+" "+str(p[2])+"\n")
    plyfile.close()


def saveInnpy(points,normals=None, gt=None, isVal=False, fname=None):
    if fname is not None:
        if isVal:
            filename = fname+"_val.npy"
        else:
            filename = fname+"_train.npy"
    else:
        filename = "val.npy" if isVal else "train.npy" 
    with open(filename,'wb') as npyfile:
        if not normals is None:
            np.save(npyfile, np.array(torch.hstack((points, normals,gt))))
        else:
            np.save(npyfile, np.array(points))
        npyfile.close()

def plotimage(outfolder, title, filename, traindata, valdata=None):
    if not os.path.exists(outfolder):
        mkdir_p(outfolder)
    if not valdata is None:
        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.suptitle(title)
        ax1.plot(traindata)
        ax2.plot(valdata)
        plt.savefig(os.path.join(outfolder, filename))
        plt.close()
    else:
        plt.suptitle(title)
        plt.plot(traindata)
        plt.savefig(os.path.join(outfolder, filename))
        plt.close()

def plotloss(outfolder, epoch, filename, traindata, valdata=None):
    out = os.path.join(outfolder, 'loss_png')
    title = 'loss - epoch ' + str(epoch) 
    name = 'loss_'+ filename +'.png'
    plotimage(out, title, name, traindata, valdata)

def plotgauss(outfolder, epoch, filename, traindata, valdata=None):
    out = os.path.join(outfolder, 'gaussavg_png')
    title = 'gaussavg - epoch ' + str(epoch) 
    name = 'gaussavg_'+ filename +'.png'
    plotimage(out, title, name, traindata, valdata)


def plotCurvature(model, filepath, resolution=128):
    mesh = o3d.io.read_triangle_mesh(filepath+'_'+str(resolution)+'.obj')
    pcl = mesh.sample_points_poisson_disk(number_of_points=int(250000))
    samples = np.asarray(pcl.points)
    implicit_curv = getImplicitCurvatureforSamples(model, samples)
    np.save(filepath+'_imp_hist_'+str(resolution)+'.npy', implicit_curv)
    hist, bins = np.histogram(implicit_curv, bins=30)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.xscale(log)
    plt.hist(implicit_curv, bins=logbins)
    plt.savefig(filepath+'_imp_hist_'+str(resolution)+'.jpg')

def getHausdorffDist(points1, points2):
 # one direction
    haus1 = Hausdorff(points1, points2)[0]
    haus2 = Hausdorff(points2, points1)[0]

    return (haus1+haus2)



def getChamferDist(points1, points2):
 # one direction
    kd_tree1 = KDTree(points1)
    dist1, _ = kd_tree1.query(points2)
    #print(dist1)
    #print(len(dist1))
    #chamfer1 = np.mean(np.square(dist1))
    #chamfer1 = np.mean(dist1)

    # other direction
    kd_tree2 = KDTree(points2)
    dist2, _ = kd_tree2.query(points1)
    #print(len(dist2))
    #chamfer2 = np.mean(np.square(dist2))
    #chamfer2 = np.mean(dist2)

    cham1 = np.sum(dist1) + np.sum(dist2)
    cham2 = np.mean(dist1) + np.mean(dist2)
    cham3 = np.sum(np.square(dist1)) + np.sum(np.square(dist2))
    cham4 = np.mean(np.square(dist1)) + np.mean(np.square(dist2))
   
    chamdist = {'sum': cham1, 'mean':cham2, 'sumsquare':cham3, 'meansquare':cham4}

    return chamdist


def clusterByNormal(fname, points, normals):
    #print("in heere", flush=True) 
    kdtree = KDTree(points)
    index = kdtree.query_ball_point(points, r=0.02)
    normals = torch.tensor(normals)
    dist = []
    percent = []
    numneigh = []
    for i in range(len(points)):
        cosinedist = 1- torch.nn.functional.cosine_similarity(normals[i].repeat(len(index[i])).reshape(-1,3), normals[index[i]], dim=-1)
        percent.append(sum(cosinedist < 0.5)/len(index[i]))
        numneigh.append(len(index[i]))
        dist.append(cosinedist)
    result = {'cosdist': dist, 'percent': np.vstack(percent), 'numneigh': np.vstack(numneigh)}
    np.save(fname, result)

