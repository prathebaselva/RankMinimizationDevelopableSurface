import os
import shutil
import time
import glob
import argparse
import math
import numpy as np
import random
#import plotly
#import plotly.figure_factory as ff
from skimage import measure
import torch
#from pymesh import obj
#import open3d
import torch.backends.cudnn as cudnn
from utils import * # normalize_pts, normalize_normals, SdfDataset, mkdir_p, isdir
from gradient import *
from curvature import *
from colorcode import *
from loadmodel import *
from mcube import *
import trimesh
#from chamfer_distance import ChamferDistance
#from mayavi import mlab
import json
#outfolder = '/mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/output/'
#outfolder = '/mnt/nfs/work1/kalo/gopalsharma/pselvaraju/DevelopSurf/output/'
outfolder = 'output/'

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)


def getmcubeGaussHistogram(gaussCurvature, points, epoch, args):
    histogram = np.histogram(list(gaussCurvature), bins=100)
    if not isdir(outfolder+'/gauss'):
        mkdir_p(outfolder+'/gauss')
    if not isdir(outfolder+'/gauss_png'):
        mkdir_p(outfolder+'/gauss_png')

    gauss_png_dir = outfolder+'/gauss_png/'
    gauss_dir = outfolder+'/gauss/'

    plt.hist(list(gaussCurvature), 100)
    plt.title('gauss_'+args.save_file_name+'_'+str(epoch))
    plt.savefig(gauss_png_dir+'gauss_'+args.save_file_name+'_'+str(epoch)+'.png')            
    plt.close()

def getGaussHistogram(gaussCurvature, points, epoch, isTrain, args, isTest=False):
    if isTrain:
        trainstr = 'train'
    elif isTest:
        trainstr = 'test'
    else:
        trainstr = 'val'
    histogram = np.histogram(list(gaussCurvature), bins=100)
    if not isdir(outfolder+'/gauss'):
        mkdir_p(outfolder+'/gauss')
    if not isdir(outfolder+'/gauss_png'):
        mkdir_p(outfolder+'/gauss_png')

    gauss_png_dir = outfolder+'/gauss_png/'
    gauss_dir = outfolder+'/gauss/'

    plt.hist(list(gaussCurvature), 100)
    plt.title(trainstr+'_gauss_'+args.save_file_name+'_'+str(epoch))
    plt.savefig(gauss_png_dir+trainstr+'_gauss_'+args.save_file_name+'_'+str(epoch)+'.png')            
    plt.close()
   
def getSDFandGaussColorcoded(points, IF, SVD, gaussCurvature, meanCurvature, outputfilename, writegauss=0):
        print("numpoint = ",len(points))
        print("minIF = {} and maxIF = {}".format(min(IF), max(IF)))
        print("minSVD = {} and maxSVD = {}".format(min(SVD), max(SVD)))
        print("mingauss = {} and maxgauss = {}".format(min(gaussCurvature), max(gaussCurvature)))

        numneg = (np.where(IF < 0)[0])
        numpos = (np.where(IF > 0)[0])
        numzero = (np.where(IF == 0)[0])
        print("IF numpos {} , numneg = {}, numzero = {}".format(len(numpos), len(numneg), len(numzero)))

        fname = "IF"
        dcolors = getSDFColors(IF)
        writePLY(outfolder, fname, points,dcolors, IF, outputfilename)

        #if writegauss:
        hcolors = getCurvatureColors(SVD)
        fname = "SVD"
        writePLY(outfolder, fname, points, hcolors, SVD, outputfilename)

        hcolors = getCurvatureColors(gaussCurvature)
        fname = "gauss"
        writePLY(outfolder, fname, points, hcolors, gaussCurvature, outputfilename)

        hcolors = getCurvatureColors(meanCurvature)
        fname = "mean"
        writePLY(outfolder, fname, points, hcolors, meanCurvature, outputfilename)

# testing function
# testing function

def getSurfaceSamplePoints(dataset, latent, model, epoch, count, args, prefname="best"):
    rotv, simplices = getmcubePoints(dataset, latent, model, args)
    
    if not len(rotv):
        return None

    mesh = trimesh.Trimesh(np.array(rotv), np.array(simplices))
    trimesh.exchange.export.export_mesh(mesh, os.path.join(outfolder,args.save_file_name+'_'+prefname+'.obj'))
    samples, _ = trimesh.sample.sample_surface_even(mesh, count)
    return samples

def getDiscreteCurvatureOfSurface(rotv, simplices, args, prefname="best", count=250000):
    mesh = trimesh.Trimesh(np.array(rotv), np.array(simplices))
    surface_area = measure.mesh_surface_area(rotv, simplices)
    trimesh.exchange.export.export_mesh(mesh, os.path.join(outfolder,args.save_file_name+'_'+prefname+'.obj'))
    samples, _ = trimesh.sample.sample_surface_even(mesh, count)
    print(len(samples))


    npoint = len(samples)
    pointindex = np.arange(npoint)
    np.random.shuffle(pointindex)
    curvature_samples = samples
    if npoint > 300000:
        curvature_samples = samples[pointindex[0:300000]]

    numsamples = len(curvature_samples)
    discrete_curv1 = np.abs(np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh=mesh, points=curvature_samples, radius=0.1)))
    wheredisccurv1_1e1 = np.where(discrete_curv1 <= 1e-1)[0]
    wheredisccurv1_1e2 = np.where(discrete_curv1 <= 1e-2)[0]
    wheredisccurv1_1e3 = np.where(discrete_curv1 <= 1e-3)[0]
    disccurv_1e1 = ((1.0 - (len(wheredisccurv1_1e1)/numsamples))) 
    disccurv_1e2 = ((1.0 - (len(wheredisccurv1_1e2)/numsamples))) 
    disccurv_1e3 = ((1.0 - (len(wheredisccurv1_1e3)/numsamples))) 
    d_disccurv_1e1 = (((len(wheredisccurv1_1e1)/numsamples))) 
    d_disccurv_1e2 = (((len(wheredisccurv1_1e2)/numsamples))) 
    d_disccurv_1e3 = (((len(wheredisccurv1_1e3)/numsamples))) 
    disccurv_median = (np.median(discrete_curv1)) 
    disccurv_mean = (np.mean(discrete_curv1)) 
    print("disccurv_1e1 = ", d_disccurv_1e1)
    print("disccurv_1e2 = ", d_disccurv_1e2)
    print("disccurv_1e3 = ", d_disccurv_1e3)
    print("disccurv_median = ", disccurv_median)
    print("disccurv_avg = ", disccurv_mean)
    d_disc_thresh = [d_disccurv_1e1, d_disccurv_1e2, d_disccurv_1e3]
    return samples, surface_area, [d_disc_thresh, None], [disccurv_mean, None], [disccurv_median, None]

def getGaussAvgOfSurfacePoints(dataset, latent, model, epoch, count, args, prefname="best"):
    rotv, simplices = getmcubePoints(dataset, latent, model, args)
    
    if not len(rotv):
        return [], 0, [1000,1000], [1000,1000],[1000,1000]

    mesh = trimesh.Trimesh(np.array(rotv), np.array(simplices))
    surface_area = measure.mesh_surface_area(rotv, simplices)
    trimesh.exchange.export.export_mesh(mesh, os.path.join(outfolder,args.save_file_name+'_'+prefname+'.obj'))
    samples, _ = trimesh.sample.sample_surface_even(mesh, count)
    print(len(samples))


    npoint = len(samples)
    pointindex = np.arange(npoint)
    np.random.shuffle(pointindex)
    curvature_samples = samples
    if npoint > 300000:
        curvature_samples = samples[pointindex[0:300000]]

    numsamples = len(curvature_samples)
    discrete_curv1 = np.abs(np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh=mesh, points=curvature_samples, radius=0.1)))
#    discrete_curv3 = np.abs(np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh=mesh, points=curvature_samples, radius=0.3)))
#    discrete_curv5 = np.abs(np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh=mesh, points=curvature_samples, radius=0.05)))
    implicit_curv = np.abs(getImplicitCurvatureforSamples(model, latent, curvature_samples, args))


    wheredisccurv1_1e1 = np.where(discrete_curv1 <= 1e-1)[0]
#    wheredisccurv3_1e1 = np.where(discrete_curv3 <= 1e-1)[0]
#    wheredisccurv5_1e1 = np.where(discrete_curv5 <= 1e-1)[0]

   
    wheredisccurv1_1e2 = np.where(discrete_curv1 <= 1e-2)[0]
#    wheredisccurv3_1e2 = np.where(discrete_curv3 <= 1e-2)[0]
#    wheredisccurv5_1e2 = np.where(discrete_curv5 <= 1e-2)[0]

    wheredisccurv1_1e3 = np.where(discrete_curv1 <= 1e-3)[0]
#    wheredisccurv3_1e3 = np.where(discrete_curv3 <= 1e-3)[0]
#    wheredisccurv5_1e3 = np.where(discrete_curv5 <= 1e-3)[0]

#    disccurv_1e1 = ((1.0 - (len(wheredisccurv1_1e1)/numsamples)) +  (1.0 - (len(wheredisccurv3_1e1)/numsamples)) + (1.0 - (len(wheredisccurv5_1e1)/numsamples)))/3
#    disccurv_1e2 = ((1.0 - (len(wheredisccurv1_1e2)/numsamples)) +  (1.0 - (len(wheredisccurv3_1e2)/numsamples)) + (1.0 - (len(wheredisccurv5_1e2)/numsamples)))/3
#    disccurv_1e3 = ((1.0 - (len(wheredisccurv1_1e3)/numsamples)) +  (1.0 - (len(wheredisccurv3_1e3)/numsamples)) + (1.0 - (len(wheredisccurv5_1e3)/numsamples)))/3

    disccurv_1e1 = ((1.0 - (len(wheredisccurv1_1e1)/numsamples))) 
    disccurv_1e2 = ((1.0 - (len(wheredisccurv1_1e2)/numsamples))) 
    disccurv_1e3 = ((1.0 - (len(wheredisccurv1_1e3)/numsamples))) 
    d_disccurv_1e1 = (((len(wheredisccurv1_1e1)/numsamples))) 
    d_disccurv_1e2 = (((len(wheredisccurv1_1e2)/numsamples))) 
    d_disccurv_1e3 = (((len(wheredisccurv1_1e3)/numsamples))) 
    #d_disccurv_1e1 = (((len(wheredisccurv1_1e1)/numsamples)) +  ((len(wheredisccurv3_1e1)/numsamples)) + ((len(wheredisccurv5_1e1)/numsamples)))/3
#    d_disccurv_1e2 = (((len(wheredisccurv1_1e2)/numsamples)) +  ((len(wheredisccurv3_1e2)/numsamples)) + ((len(wheredisccurv5_1e2)/numsamples)))/3
#    d_disccurv_1e3 = (((len(wheredisccurv1_1e3)/numsamples)) +  ((len(wheredisccurv3_1e3)/numsamples)) + ((len(wheredisccurv5_1e3)/numsamples)))/3
    
    whereimpcurv_1e1 = np.where(implicit_curv <= 1e-1)[0]
    whereimpcurv_1e3 = np.where(implicit_curv <= 1e-3)[0]
    whereimpcurv_1e2 = np.where(implicit_curv <= 1e-2)[0]

    impcurv_1e1 = (1.0 - (len(whereimpcurv_1e1)/numsamples))
    impcurv_1e3 = (1.0 - (len(whereimpcurv_1e3)/numsamples))
    impcurv_1e2 = (1.0 - (len(whereimpcurv_1e2)/numsamples))

    i_impcurv_1e1 = ((len(whereimpcurv_1e1)/numsamples))
    i_impcurv_1e3 = ((len(whereimpcurv_1e3)/numsamples))
    i_impcurv_1e2 = ((len(whereimpcurv_1e2)/numsamples))

    #histogram_disc = np.histogram(list(discrete_curv3), bins=100)
    #histogram_imp = np.histogram(list(implicit_curv), bins=100)
    #if epoch % 3 == 0:
    #    print(histogram_disc)
    #    print("-------")
    #    print(histogram_imp)

    disccurv_median = (np.median(discrete_curv1)) 
    #disccurv_median = (np.median(discrete_curv1) + np.median(discrete_curv3) + np.median(discrete_curv5))/3
    #disccurv_mean = (np.mean(discrete_curv1) + np.mean(discrete_curv3) + np.mean(discrete_curv5))/3
    disccurv_mean = (np.mean(discrete_curv1)) 

    impcurv_median = np.median(implicit_curv)
    impcurv_mean = np.mean(implicit_curv)

    print("disccurv_1e1 = ", d_disccurv_1e1)
    print("disccurv_1e2 = ", d_disccurv_1e2)
    print("disccurv_1e3 = ", d_disccurv_1e3)
    print("disccurv_median = ", disccurv_median)
    print("disccurv_avg = ", disccurv_mean)
    print("-------------------------------------------")
    print("impcurv_1e1 = ", i_impcurv_1e1)
    print("impcurv_1e2 = ", i_impcurv_1e2)
    print("impcurv_1e3 = ", i_impcurv_1e3)
    print("impcurv_median = ", impcurv_median)
    print("impcurv_avg = ", impcurv_mean)

    d_disc_thresh = [d_disccurv_1e1, d_disccurv_1e2, d_disccurv_1e3]
    i_imp_thresh = [i_impcurv_1e1, i_impcurv_1e2, i_impcurv_1e3]
    return samples, surface_area, [d_disc_thresh, i_imp_thresh], [disccurv_mean, impcurv_mean], [disccurv_median, impcurv_median]

def getDiscAndImpCurvatureOfSurface(dataset, latent, model, epoch, count, args, prefname="best"):
    samples, surface_area, threshcurv, meancurv, mediancurv = getGaussAvgOfSurfacePoints(dataset, latent, model, epoch, count, args, prefname)

    if len(samples) == 0:
        return [], 0,1000,1000,1000
   
    return samples, surface_area, threshcurv, meancurv, mediancurv


def getColorCurvatureOfSurface(dataset, latent, model, epoch, count, args, prefname="best"):
    samples = getSurfaceSamplePoints(dataset, latent, model, epoch, count, args, prefname)

    if len(samples) == 0:
        return
   
    #points, IF, SVD, gaussCurvature, meanCurvature = getSDFandCurvatureforSamples(model, latent, samples, args)
    points, gaussCurvature = getGaussCurvatureforSamples(model, latent, samples, args)
    writeGaussPLYPts(outfolder, 'gauss', points, gaussCurvature, prefname)
    #rotv, simplices = getmcubePoints(dataset, latent, model, args)
    
    #if not len(rotv):
    #    return [], 0, [1000,1000],[1000,1000]

    #mesh = trimesh.Trimesh(np.array(rotv), np.array(simplices))
    #surface_area = measure.mesh_surface_area(rotv, simplices)
    #trimesh.exchange.export.export_mesh(mesh, os.path.join(outfolder,args.save_file_name+'_'+prefname+'.obj'))
    #print("min and max of mcube IF = {} {}".format(IF.min(), IF.max()))
    #print("abs min and max of mcube IF = {} {}".format(np.abs(IF).min(), np.abs(IF).max()))

    #if count <= 100000:
    #getSDFandGaussColorcoded(points, IF, SVD, gaussCurvature, meanCurvature, prefname, args.gauss)
    return points, gaussCurvature
    #else:
    #    r = np.random.randint(len(samples), size=100000)
    #    getSDFandGaussColorcoded(points[r], IF[r], SVD[r], gaussCurvature[r], None, args.save_file_name, args.gauss)
    #return samples, gauss,  surface_area, IF
    #return samples, surface_area, threshcurv, meancurv, mediancurv
#def test(dataset, model, args, fname):
#    getSurfacePoints(dataset, model, 0, 100000, args,'test')

def test(dataset, model, args, fname):
    #normalize gt_point cloud
    gt_pointcloud = np.loadtxt(args.gt_pts)
    gt_points = normalize_pts_withdia(gt_pointcloud[:, :3])

    pre_model = initModel(args)
    cudnn.benchmark = True
    pre_model = loadPretrainedModel(pre_model, args)
    pre_model.to(device)

    gt_gausssamples = gt_points

    gt_gauss_reg = abs(getImplicitCurvatureforSamples(model, gt_gausssamples, args))
    gt_gauss_pre = abs(getImplicitCurvatureforSamples(pre_model, gt_gausssamples, args))

    print("reg_gt_median = ",np.median(gt_gauss_reg))
    print("reg_gt_avg = ",np.mean(gt_gauss_reg))
    print("pre_gt_median = ",np.median(gt_gauss_pre))
    print("pre_gt_avg = ",np.mean(gt_gauss_pre))

    rotv, simplices = getmcubePoints(dataset, model, args)
    pre_rotv, pre_simplices = getmcubePoints(dataset, pre_model, args)
    
    if not len(rotv):
        return [], 0, 1000,1000,1000,1000

    mesh = trimesh.Trimesh(np.array(rotv), np.array(simplices))
    #trimesh.exchange.export.export_mesh(mesh, os.path.join(outfolder,args.save_file_name+'_test.obj'))
    samples, _ = trimesh.sample.sample_surface_even(mesh, 400000)


    npoint = len(samples)
    pointindex = np.arange(npoint)
    np.random.shuffle(pointindex)
    pred_gausssamples = samples
    if npoint > 250000:
        pred_gausssamples = samples[pointindex[0:250000]]
    pred_gauss = abs(getImplicitCurvatureforSamples(model, pred_gausssamples, args))
    print(" numpoints = ", len(pred_gausssamples))
    print("reg_median = ",np.median(pred_gauss))
    print("reg_avg = ",np.mean(pred_gauss))

    pre_mesh = trimesh.Trimesh(np.array(pre_rotv), np.array(pre_simplices))
    #trimesh.exchange.export.export_mesh(mesh, os.path.join(outfolder,args.save_file_name+'_test.obj'))
    pre_samples, _ = trimesh.sample.sample_surface_even(pre_mesh, 400000)
    npoint = len(pre_samples)
    pointindex = np.arange(npoint)
    np.random.shuffle(pointindex)
    p_gausssamples = pre_samples
    if npoint > 250000:
        p_gausssamples = pre_samples[pointindex[0:250000]]
    p_gauss = abs(getImplicitCurvatureforSamples(model, p_gausssamples, args))
    pretrained_gauss = abs(getImplicitCurvatureforSamples(pre_model, p_gausssamples, args))
    print("pre for numpoints = ", len(pred_gausssamples))
    print("pre median = ",np.median(pretrained_gauss))
    print("pre avg = ",np.mean(pretrained_gauss))
    print("pre for reg median = ",np.median(p_gauss))
    print("pre for reg avg = ",np.mean(p_gauss),flush=True)
    
    #gt_1 = np.where(gt_gauss <= 10)[0]
    #pred_1 = np.where(pred_gauss <= 10)[0]

    # one direction
    kd_tree1 = KDTree(gt_gausssamples)
    distreg1, KNNreg1 = kd_tree1.query(pred_gausssamples)
    distpre1, KNNpre1 = kd_tree1.query(p_gausssamples)

    # other direction
    #kd_tree2 = KDTree(pred_gausssamples)
    #dist2, KNN2 = kd_tree2.query(gt_gausssamples)

    #kd_tree3 = KDTree(p_gausssamples)
    #dist3, KNN3 = kd_tree3.query(gt_gausssamples)
    
    gaussdiff1 = np.mean(abs(gt_gauss_reg - pred_gauss[KNNreg1]))
    gaussdiff2 = np.mean(abs(gt_gauss_pre - p_gauss[KNNpre1]))
    #gaussdiff3 = np.mean(abs(pred_gauss - gt_gauss[KNN2]))
    #gaussdiff4 = np.mean(abs(pred_gauss - p_gauss[KNN3]))

    #chamfer1 = np.mean(np.square(dist1))
    #chamfer2 = np.mean(np.square(dist2))
    print("gauss 1= ",gaussdiff1)
    print("gauss 2= ",gaussdiff2)
    #print("gauss = ",gaussdiff1 + gaussdiff2)
    #print("chamfer = ",chamfer1 + chamfer2)



def comparesdfthreshold(mcube_points1, mcube_points2, num_batch, model, args):
    avgsdf = 0
    surfaceavgsdf = 0
    count = 0
    histmid = []
    model.eval()

    histbins = np.array([0.1**a for a in range(32)])
    histbins = np.insert(histbins, 0, 100)
    histbins = np.append(histbins, 0)
    histbins = np.flip(histbins)
    histcomp = []
    for mcube_points in [mcube_points1, mcube_points2]:
        for i in range(num_batch):
            r = np.random.randint(len(mcube_points), size=(args.train_batch,))
            mcube_sampled_points =  torch.tensor(mcube_points[r]).to(device)
            mcube_predicted_sdf = model(mcube_sampled_points.float())
                
            denom = 10**(torch.floor(torch.log(mcube_predicted_sdf)/torch.log(torch.tensor(10))))
            histogram = np.histogram(list(np.abs(denom.clone().detach().cpu().numpy())), bins=histbins)
            if len(histmid):
                histmid += (np.array(histogram[0]) *np.array(histogram[1][1:]))
            else:
                histmid = np.array(histogram[0]) * np.array(histogram[1][1:])
        histcomp.append(histmid)    
    diff = np.sum(histcomp[0] - histcomp[1])
    return diff

    

def getavgsdfthreshold(mcube_points, num_batch, model, args):
    avgsdf = 0
    surfaceavgsdf = 0
    count = 0
    histmid = []
    model.eval()

    histbins = np.array([0.1**a for a in range(32)])
    histbins = np.insert(histbins, 0, 100)
    histbins = np.append(histbins, 0)
    histbins = np.flip(histbins)

    for i in range(num_batch):
        r = np.random.randint(len(mcube_points), size=(args.train_batch,))
        mcube_sampled_points =  torch.tensor(mcube_points[r]).to(device)
        mcube_predicted_sdf = model(mcube_sampled_points.float())
            
        denom = 10**(torch.floor(torch.log(mcube_predicted_sdf)/torch.log(torch.tensor(10))))
        histogram = np.histogram(list(np.abs(denom.clone().detach().cpu().numpy())), bins=histbins)
        if len(histmid):
            histmid += np.array(histogram[0])
        else:
            histmid = np.array(histogram[0])

    print(histmid)
    print(np.max(histmid))
    print(np.argmax(histmid))
    threshold = (histbins[np.argmax(histmid)])
    #threshold = (histbins[np.argmax(histmid)] + histbins[np.argmax(histmid)+1])/2
    print("threshold = ", threshold)

    #return avgsdf/count, histmid/num_batch #surfaceavgsdf/count
    return threshold#surfaceavgsdf/count



def getGaussAvgOfSurfacePointsbkup(dataset, model, epoch, count, args, prefname="best"):
    rotv, simplices = getmcubePoints(dataset, model, args)
    
    if not len(rotv):
        return [], 0, 1000,1000,1000,1000

    mesh = trimesh.Trimesh(np.array(rotv), np.array(simplices), encoding='ascii')

    surface_area = measure.mesh_surface_area(rotv, simplices)
    trimesh.exchange.export.export_mesh(mesh, os.path.join(outfolder,args.save_file_name+'_'+prefname+'.obj'))
    #pmesh = obj.Obj(os.path.join(outfolder, args.save_file_name+'_'+prefname+'.obj'))
    #pymesh.add_attribute("vertex_gaussian_curvature")
    #pgauss = pmesh.get_attribute("vertex_gaussian_curvature")
    #pgauss = np.abs(pgauss).mean()
 
   
    #if not args.phase == "test": 
    print("count = ", count)
    samples, _ = trimesh.sample.sample_surface_even(mesh, count)

    gaussvert = np.abs(np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh=mesh, points=mesh.vertices, radius=0)))
    gausspoint = np.abs(np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh=mesh, points=samples, radius=0.05)))
   
    numgaussvert1e3 = np.where(gaussvert <= 1e-3)[0]
    numgausspoint1e3 = np.where(gausspoint <= 1e-3)[0]

    numgaussvert1 = np.where(gaussvert <= 1)[0]
    numgausspoint1 = np.where(gausspoint <= 1)[0]

    histogram1 = np.histogram(list(gaussvert), bins=100)
    histogram2 = np.histogram(list(gausspoint), bins=100)
    if epoch % 3 == 0:
        print(histogram1)
        print("-------")
        print(histogram2)
    #getmcubeGaussHistogram(gauss, samples, epoch, args)
    #gauss = len(gauss)* gauss[numgauss0].mean()
    #gauss1 = (1.0 - (len(numgauss1)/len(samples))) *gauss[numgauss1].mean()
    #gauss5 = (1.0 - (len(numgauss5)/len(samples))) *gauss[numgauss5].mean()
    numvertices = len(mesh.vertices)
    gaussvert1e3 = (1.0 - (len(numgaussvert1e3)/numvertices))
    gausspoint1e3 = (1.0 - (len(numgausspoint1e3)/len(samples)))

    gaussvert1 = (1.0 - (len(numgaussvert1)/numvertices))
    gausspoint1 = (1.0 - (len(numgausspoint1)/len(samples)))

    meangaussvert = gaussvert[numgaussvert1].mean()
    meangausspoint = gausspoint[numgausspoint1].mean()

    mediangaussvert = np.median(gaussvert)
    mediangausspoint = np.median(gausspoint)
    
    #print("gauss curvature of the mesh = ",gauss)
    #print("pymesh gauss curvature of the mesh = ",pgauss)
    return samples, surface_area, [gaussvert1e3, gausspoint1e3], [gaussvert1, gausspoint1], [meangaussvert, meangausspoint], [mediangaussvert, mediangausspoint]


