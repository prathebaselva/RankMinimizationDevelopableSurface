import open3d as o3d
import numpy as np
import trimesh
import torch
from .mcube import getmcubePoints, getImplicitCurvatureforSamples
import matplotlib.pyplot as plt



def plotImplicitCurvature(implicit_gausscurv, implicit_mincurv, implicit_maxcurv, implicit_det, args):
    implicit_gausscurv = implicit_gausscurv.detach().cpu().numpy()
    implicit_mincurv = implicit_mincurv.detach().cpu().numpy()
    implicit_maxcurv = implicit_maxcurv.detach().cpu().numpy()
    implicit_det = implicit_det.detach().cpu().numpy()
    np.save(args.save_file_path+'_impgauss_hist.npy', implicit_gausscurv)
    np.save(args.save_file_path+'_impmin_hist.npy', implicit_mincurv)
    np.save(args.save_file_path+'_impmax_hist.npy', implicit_maxcurv)
    np.save(args.save_file_path+'_impdet_hist.npy', implicit_det)
#    hist, bins = np.histogram(implicit_curv, bins=20, range=(1e-10, 1e6))
#    print(np.mean(implicit_curv))
#    print(np.median(implicit_curv))
#    bins[np.where(bins == 0)] = 1e-7
#    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
#    plt.xscale('log')
#    plt.hist(implicit_curv, bins=logbins, color="skyblue", ec="black")
#    plt.savefig(args.save_file_path+'_imp_hist.jpg')
#    plt.close()

def plotImplicitCurvatureFromSamples(gridsamples, modelnet, args, resolution=128, latent=None, filename=None):
    rotv, simplices = getmcubePoints(gridsamples, modelnet, args, latent)
    #print(len(rotv))
    if len(rotv) > 0:
        if filename is None:
            trimesh.Trimesh(np.array(rotv), np.array(simplices)).export(args.save_file_path+'_'+str(resolution)+'.obj')
            mesh = o3d.io.read_triangle_mesh(args.save_file_path+'_'+str(resolution)+'.obj')
        else:
            trimesh.Trimesh(np.array(rotv), np.array(simplices)).export(filename+'_'+str(resolution)+'.obj')
            mesh = o3d.io.read_triangle_mesh(filename+'_'+str(resolution)+'.obj')
        #pcl = mesh.sample_points_poisson_disk(number_of_points=int(60000))
        #samples = np.asarray(pcl.points)
        #print("sampled")
        #print(np.mean(samples))
        #print(np.std(samples))
        return
        implicit_curv = getImplicitCurvatureforSamples(modelnet, samples, args)
        np.save(args.save_file_path+'_imp_hist_'+str(resolution)+'.npy', implicit_curv)
        hist, bins = np.histogram(implicit_curv, bins=30)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.xscale('log')
        plt.hist(implicit_curv, bins=logbins)
        plt.savefig(args.save_file_path+'_imp_hist_'+str(resolution)+'.jpg')
        plt.close()
    return None

