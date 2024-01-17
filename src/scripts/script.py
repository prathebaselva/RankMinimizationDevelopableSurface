import os
import torch
import errno
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d
import trimesh
from trimeshcurvature import discrete_mean_curvature_measure, discrete_gaussian_curvature_measure
from gradient import getGradientAndHessian
from test import Tester
from curvature import gaussianCurvature
from mcube import getmcubePoints
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import directed_hausdorff as Hausdorff
matplotlib.use("Agg")

def curvatureHistogram(npyfile, filepath, threshold):
    impcurv = np.load(npyfile)
    hist, bins = np.histogram(impcurv, bins=20, range=(1e-15,threshold))
    #hist, bins = np.histogram(impcurv, bins=30, range=(1e-7,1))
    bins[np.where(bins == 0)] = 1e-15
    plt.figure(figsize=(10,10))
    x, y, _ = plt.hist(impcurv, bins=bins, color='skyblue', ec='black')
    plt.savefig(filepath+'_imp_hist.png' )
    plt.close()

    hist, bins = np.histogram(impcurv, bins=20, range=(1e-15, threshold))
    bins[np.where(bins == 0)] = 1e-15
    #hist, bins = np.histogram(impcurv, bins=30, range=(1e-7, 1))
    #print(np.median(hist))
    #print(hist)
    #print(bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    #print(logbins)

    #ind = np.searchsorted(logbins, hist, side='right')
    #m = [np.median(hist[ind == label]) for label in range(logbins.size - 1)]
    #m1 = [np.mean(hist[ind == label]) for label in range(logbins.size - 1)]
    #print(m)
    #print(m1)

    #print(pd.Series(hist).groupby(pd.cut(hist, logbins)).median())
    #print(pd.Series(hist).groupby(pd.cut(hist, bins)).median())
    plt.figure(figsize=(10,10))
    x, y, _ = plt.hist(impcurv, bins=logbins, color='skyblue', ec='black', log=True)
    plt.xscale('log')
    plt.savefig(filepath+'_imp_hist_log.png')
    plt.close()

    implicit_curv = impcurv
    numsamples = len(impcurv)
    print(numsamples)

#    whereimpcurv_gt1e9 = np.where(implicit_curv >= 1e9)[0]
#    whereimpcurv_gt1e7 = np.where(implicit_curv >= 1e7)[0]
#    whereimpcurv_gt1e5 = np.where(implicit_curv >= 1e5)[0]
#    whereimpcurv_gt1e4 = np.where(implicit_curv >= 1e4)[0]
    whereimpcurv_1 = np.where(implicit_curv <= 1)[0]
    whereimpcurv_1000 = np.where(implicit_curv <= 1000)[0]
    whereimpcurv_100 = np.where(implicit_curv <= 100)[0]
    whereimpcurv_10 = np.where(implicit_curv <= 10)[0]
    whereimpcurv_1 = np.where(implicit_curv <= 1)[0]
    whereimpcurv_1e1 = np.where(implicit_curv <= 1e-1)[0]
    whereimpcurv_1e3 = np.where(implicit_curv <= 1e-3)[0]
    whereimpcurv_1e5 = np.where(implicit_curv <= 1e-5)[0]
    whereimpcurv_1e7 = np.where(implicit_curv <= 1e-7)[0]

    impcurv_1000 = (1.0 - (len(whereimpcurv_1000)/numsamples))
    impcurv_100 = (1.0 - (len(whereimpcurv_100)/numsamples))
    impcurv_10 = (1.0 - (len(whereimpcurv_10)/numsamples))
    impcurv_1 = (1.0 - (len(whereimpcurv_1)/numsamples))
    impcurv_1e1 = (1.0 - (len(whereimpcurv_1e1)/numsamples))
    impcurv_1e3 = (1.0 - (len(whereimpcurv_1e3)/numsamples))
    impcurv_1e5 = (1.0 - (len(whereimpcurv_1e5)/numsamples))
    impcurv_1e7 = (1.0 - (len(whereimpcurv_1e7)/numsamples))
#    impcurv_gt1e4 = (1.0 - (len(whereimpcurv_gt1e4)/numsamples))
#    impcurv_gt1e5 = (1.0 - (len(whereimpcurv_gt1e5)/numsamples))
#    impcurv_gt1e7 = (1.0 - (len(whereimpcurv_gt1e7)/numsamples))
#    impcurv_gt1e9 = (1.0 - (len(whereimpcurv_gt1e9)/numsamples))
#

    i_impcurv_1000 = ((len(whereimpcurv_1000)/numsamples))
    i_impcurv_100 = ((len(whereimpcurv_100)/numsamples))
    i_impcurv_10 = ((len(whereimpcurv_10)/numsamples))
    i_impcurv_1 = ((len(whereimpcurv_1)/numsamples))
    i_impcurv_1e1 = ((len(whereimpcurv_1e1)/numsamples))
    i_impcurv_1e3 = ((len(whereimpcurv_1e3)/numsamples))
    i_impcurv_1e5 = ((len(whereimpcurv_1e5)/numsamples))
    i_impcurv_1e7 = ((len(whereimpcurv_1e7)/numsamples))
#    i_impcurv_gt1e4 = ((len(whereimpcurv_gt1e4)/numsamples))
#    i_impcurv_gt1e5 = ((len(whereimpcurv_gt1e5)/numsamples))
#    i_impcurv_gt1e7 = ((len(whereimpcurv_gt1e7)/numsamples))
#    i_impcurv_gt1e9 = ((len(whereimpcurv_gt1e9)/numsamples))

    #impcurv_median = np.median(hist)
    impcurv_median1 = np.median(implicit_curv)
    impcurv_mean1 = np.mean(implicit_curv)

    print("impcurv_1000 = ", i_impcurv_1000)
    print("impcurv_100 = ", i_impcurv_100)
    print("impcurv_10 = ", i_impcurv_10)
    print("impcurv_1 = ", i_impcurv_1)
    print("impcurv_1e1 = ", i_impcurv_1e1)
    print("impcurv_1e3 = ", i_impcurv_1e3)
    print("impcurv_1e5 = ", i_impcurv_1e5)
    print("impcurv_1e7 = ", i_impcurv_1e7)
#    print("impcurv_gt1e4 = ", i_impcurv_gt1e4)
#    print("impcurv_gt1e5 = ", i_impcurv_gt1e5)
#    print("impcurv_gt1e7 = ", i_impcurv_gt1e7)
#    print("impcurv_gt1e9 = ", i_impcurv_gt1e9)
    print("impcurv_median = ", impcurv_median1)
    print("impcurv_avg = ", impcurv_mean1)

#filenames = '../data/testing250data.npy'
def Normalize(vertices):
    maxv = np.max(vertices, axis=0)
    minv = np.min(vertices, axis=0)
    diff = maxv - minv

    m = max(diff)/2
    c = (maxv+minv)/2
    newdata = vertices - c
    newdata = newdata/m
    return newdata

def poissonsampling(filepath, nsamples, inputfolder):
    #filename = filepath.split('/')[-1].split('.')[0]
    filename = filepath[:-4]
    objfile = os.path.join(inputfolder, filepath)
    mesh = o3d.io.read_triangle_mesh(objfile)
    pcl = mesh.sample_points_poisson_disk(number_of_points=int(nsamples))
    points = np.asarray(pcl.points)
    return points

def getHauss(mesh1, gtmesh):
    mesh = o3d.io.read_triangle_mesh(mesh1)
    pcl = mesh.sample_points_poisson_disk(number_of_points=250000)
    samples1 = np.asarray(pcl.points)
    maxv = np.max(samples1, axis=0)
    minv = np.min(samples1, axis=0)
    diff = maxv-minv
    m = max(diff)/2
    c = (maxv+minv)/2
    samples1 -= c
    samples1 /= m
    gtpoints = np.load(gtmesh)
    samples2 = (np.asarray(gtpoints[:,0:3])/2)
    maxv = np.max(samples2, axis=0)
    minv = np.min(samples2, axis=0)
    diff = maxv-minv
    m = max(diff)/2
    samples2 -= c
    samples2 /= m
    from simpleicp import PointCloud, SimpleICP

    pc_fix = PointCloud(samples1, columns=["x", "y", "z"])
    pc_mov = PointCloud(samples2, columns=["x", "y", "z"])

    # Create simpleICP object, add point clouds, and run algorithm!
    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=10.0, max_iterations=200)
    print(X_mov_transformed.shape)

    #trimesh.points.PointCloud(samples1).export('samples1.ply') 
    #trimesh.points.PointCloud(samples2).export('samples2.ply') 
    haus1 = Hausdorff(samples1, samples2)[0]
    haus2 = Hausdorff(samples2, samples1)[0]

    print(haus1+haus2, flush=True)

def getHauss1(mesh1, gtmesh):
    mesh = o3d.io.read_triangle_mesh(mesh1)
    pcl = mesh.sample_points_poisson_disk(number_of_points=250000)
    samples1 = np.asarray(pcl.points)
    mesh = o3d.io.read_triangle_mesh(gtmesh)
    pcl = mesh.sample_points_poisson_disk(number_of_points=250000)
    samples2 = np.asarray(pcl.points)
    trimesh.points.PointCloud(samples1).export('samples11.ply') 
    trimesh.points.PointCloud(samples2).export('samples21.ply') 
    haus1 = Hausdorff(samples1, samples2)[0]
    haus2 = Hausdorff(samples2, samples1)[0]

    print(haus1+haus2)

def getHausdorffDist(points1, points2):
 # one direction
    haus1 = Hausdorff(points1, points2)[0]
    haus2 = Hausdorff(points2, points1)[0]

    print(haus1+haus2)

def getCham1(mesh1, gtmesh):
    mesh = o3d.io.read_triangle_mesh(mesh1)
    pcl = mesh.sample_points_poisson_disk(number_of_points=250000)
    samples1 = np.asarray(pcl.points)
    mesh = o3d.io.read_triangle_mesh(gtmesh)
    pcl = mesh.sample_points_poisson_disk(number_of_points=250000)
    samples2 = np.asarray(pcl.points)
    #from chamferdist import ChamferDistance
    #chamferDist = ChamferDistance()
    #dist_bidirectional = chamferDist(samples1, samples2, bidirectional=True)
    #dist_bidirectional = chamferDist(torch.tensor(samples1).unsqueeze(0).float(), torch.tensor(X_mov_transformed).unsqueeze(0).float(), bidirectional=True)
    #print(dist_bidirectional, flush=True)
    getChamferDist(samples1, samples2)

def getCham(reconmesh, gtmesh, numsamples=0):
    if numsamples == 0:
        mesh = trimesh.load(reconmesh)
        mesh.remove_degenerate_faces() 
        mesh.remove_unreferenced_vertices()
        samples1 = Normalize(mesh.vertices)
        newgtmesh = trimesh.load(gtmesh)
        newgtmesh.remove_degenerate_faces() 
        newgtmesh.remove_unreferenced_vertices()
        samples2 = Normalize(newgtmesh.vertices)
    else:
        mesh = o3d.io.read_triangle_mesh(reconmesh)
        pcl = mesh.sample_points_poisson_disk(number_of_points=numsamples)
        samples1 = np.asarray(pcl.points)
        maxv = np.max(samples1, axis=0)
        minv = np.min(samples1, axis=0)
        diff = maxv-minv
        m = max(diff)/2
        c = (maxv+minv)/2
        samples1 -= c
        samples1 /= m

        ext = os.path.splitext(gtmesh)[-1]
        if ext == '.obj':
            mesh = o3d.io.read_triangle_mesh(gtmesh) 
            pcl = mesh.sample_points_poisson_disk(number_of_points=numsamples)
            gtpoints = np.asarray(pcl.points)
        else:
            gtpoints = np.load(gtmesh, allow_pickle=True)
        samples2 = (np.asarray(gtpoints[:,0:3])/2)
        maxv = np.max(samples2, axis=0)
        minv = np.min(samples2, axis=0)
        diff = maxv-minv
        m = max(diff)/2
        c1 = (maxv+minv)/2
        samples2 -= c1
        samples2 /= m

    from simpleicp import PointCloud, SimpleICP

    pc_fix = PointCloud(samples1, columns=["x", "y", "z"])
    pc_mov = PointCloud(samples2, columns=["x", "y", "z"])

    # Create simpleICP object, add point clouds, and run algorithm!
    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=10, max_iterations=200)
    print(X_mov_transformed.shape)
    #samples3 = samples2
    #samples4= samples2
    trimesh.points.PointCloud(samples1).export('samples1.ply') 
    trimesh.points.PointCloud(X_mov_transformed).export('samples2.ply') 
    #samples2[:,0] += 0.1
    #trimesh.points.PointCloud(samples2).export('samples3.ply') 
    #samples2[:,0] -= 0.2
    #trimesh.points.PointCloud(samples2).export('samples4.ply') 
    #r = np.random.randint(250000, size=(60000))
    #from chamferdist import ChamferDistance
    #chamferDist = ChamferDistance()
    #dist_bidirectional = chamferDist(torch.tensor(samples1), torch.tensor(X_mov_transformed), bidirectional=True)
    #dist_bidirectional = chamferDist(torch.tensor(samples1).unsqueeze(0).float(), torch.tensor(X_mov_transformed).unsqueeze(0).float(), bidirectional=True)
    #print(dist_bidirectional, flush=True)
    #dist_forward = chamferDist(samples1, X_mov_transformed)
    #dist_backward = chamferDist(X_mov_transformed, samples1)
    getChamferDist(samples1, X_mov_transformed)

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
    cham4 = cham3 / 500000 # np.meannp.square(dist1) + np.mean(np.square(dist2))
    #cham3 = np.sum(np.square(dist1)) + np.sum(np.square(dist2))
    #cham4 = np.mean(np.square(dist1) + np.mean(np.square(dist2))
   
    chamdist = {'sum': cham1, 'mean':cham2, 'sumsquare':cham3, 'meansquare':cham4}
    print(chamdist, flush=True)

    return chamdist


def getColorCodedDeterminant(model, samples, filename):
    from colorcode import getDeterminantColorcoded
    print(len(samples), flush=True)
    gaussCurvature = np.zeros((len(samples), ))
    determinant = np.zeros(len(samples), )
    model.eval()
    start_idx = 0
    numbatch = int(np.ceil(len(samples)/1024))
    print("numbatch sdf and curvature = ", numbatch)
    for i in range(numbatch):
        end = np.min((len(samples), (i+1)*1024))
        #print("{} {}".format(i*args.train_batch, end))
        xyz_tensor = torch.tensor(samples[i*1024:end]).to('cuda')
        this_bs = xyz_tensor.shape[0]
        xyz_tensor.requires_grad = True
        pred_sdf_tensor = model.run(xyz_tensor.float())
        predicted_gradient, hessian_matrix = getGradientAndHessian(pred_sdf_tensor, xyz_tensor)
        det, gauss = gaussianCurvature(predicted_gradient, hessian_matrix)
        end_idx = start_idx + this_bs
        gaussCurvature[start_idx:end_idx] = gauss.data.cpu().numpy()
        determinant[start_idx:end_idx] = det.cpu().numpy()
        start_idx = end_idx

    getDeterminantColorcoded(samples, determinant, filename)
    return gaussCurvature

def getColorCodedDeterminantforSamples(samplefile,  determinantfile, filename):
    from colorcode import getDeterminantColorcoded
    samples = np.load(samplefile)
    determinant = np.load(determinantfile)
    getDeterminantColorcoded(samples, determinant, filename)

def getColorCodedGaussforSamples(samplefile,  gaussfile, filename):
    from colorcode import getGaussColorcoded
    samples = np.load(samplefile)
    gauss = np.load(gaussfile)
    getGaussColorcoded(samples, gauss, filename)

def getColorCodedVarforSamples(samplefile,  varfile, filename):
    from colorcode import getVarianceColorcoded
    samples = np.load(samplefile)
    var = np.load(varfile)
    distfile = re.sub('_var_','_dist_', varfile)
    dist = np.load(distfile)
    print(dist[0:10])
    exit()
    getVarianceColorcoded(samples, var, filename)

def getImplicitCurvatureforSamples(model, samples):
    print(len(samples), flush=True)
    gaussCurvature = np.zeros((len(samples), ))
    model.eval()
    start_idx = 0
    numbatch = int(np.ceil(len(samples)/1024))
    print("numbatch sdf and curvature = ", numbatch)
    for i in range(numbatch):
        end = np.min((len(samples), (i+1)*1024))
        #print("{} {}".format(i*args.train_batch, end))
        xyz_tensor = torch.tensor(samples[i*1024:end]).to('cuda')
        this_bs = xyz_tensor.shape[0]
        xyz_tensor.requires_grad = True
        pred_sdf_tensor = model.run(xyz_tensor.float())
        predicted_gradient, hessian_matrix = getGradientAndHessian(pred_sdf_tensor, xyz_tensor)
        _, gauss = gaussianCurvature(predicted_gradient, hessian_matrix)
        end_idx = start_idx + this_bs
        gaussCurvature[start_idx:end_idx] = gauss.data.cpu().numpy()
        start_idx = end_idx

    return gaussCurvature

def compareVertexQuality(plyfile):
    fread = open(plyfile, 'r')
    curv = []
    for line in fread:
        data = line.strip().split()
        if len(data) == 8:
            curv.append(np.abs(float(data[-1])))
    curv = np.array(curv)
    print(np.mean(curv))
    print(np.median(curv))
    print(len(np.where(curv < 1e-3)[0]))

def getDiscreteCurvatureSurfacePoints(objfile, numsamples, scale = 1, radius=[0]):
    fname = objfile.split('/')[-1]
    fname = fname.split('.')[0]
    #numsamples = 60000
    mesh = trimesh.load(objfile)
    mesh.remove_degenerate_faces() 
    mesh.remove_unreferenced_vertices()
    newvertices = Normalize(mesh.vertices)

    #mesh = trimesh.Trimesh(vertices = mesh.vertices * scale, faces=mesh.faces)
    mesh = trimesh.Trimesh(vertices = newvertices, faces=mesh.faces)
    #mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_degenerate_faces() 
    mesh.remove_unreferenced_vertices()
    #mesh.export('recon_'+fname+'_'+str(scale)+'.obj')
    #mesh1 = o3d.io.read_triangle_mesh('recon_'+fname+'_'+str(scale)+'.obj')
    #pcl = mesh1.sample_points_poisson_disk(number_of_points=int(numsamples))
    #samples = np.asarray(pcl.points)
    
    #mesh = trimesh.load('recon_'+fname+'_'+str(scale)+'.obj')
    #print("test", flush=True)
    #samples,_ = trimesh.sample.sample_surface_even(mesh, numsamples)
    #print("test1", flush=True)
    #vertex_faces = mesh.vertex_faces
    #area_faces = mesh.area_faces
    #print(len(vertex_faces), flush=True)
    #total_area = []
    #for vf in vertex_faces:
        #print(vf)
        #print(vf[vf != -1])
    #    area = area_faces[vf[vf != -1]].sum()
    #    total_area.append(area)
        #print(area)
    #print(total_area, flush=True)
    for r in radius:
        if r == 0:
            r0samples = np.asarray(mesh.vertices) 
            numsamples = len(r0samples)
            discrete_curv1 = np.abs(np.array(discrete_gaussian_curvature_measure(mesh=mesh, points=r0samples, radius=r)))
            #print(discrete_curv1, flush=True)
            #discrete_mean_curv1 = np.array(discrete_mean_curvature_measure(mesh=mesh, points=r0samples, radius=r))
            #print(discrete_mean_curv1, flush=True)
            #sqrtterm = np.sqrt(discrete_mean_curv1**2 - discrete_curv1)
            #k1 = discrete_mean_curv1 + sqrtterm
            #k2 = discrete_mean_curv1 - sqrtterm
            #print(k1)
            #print(k2)
            #mink = np.min((k1,k2),axis=0)
            #maxk = np.max((k1,k2),axis=0)
            #print("disccurv_min_median = ", np.median(mink))
            #print("disccurv_min_avg = ", np.mean(mink), flush=True)
        else:
            numsamples = len(samples)
            discrete_curv1 =(np.abs(np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh=mesh, points=samples, radius=r))))
        #print(len(discrete_curv1))
        #exit()
        #v = np.abs(np.array(trimesh.curvature.vertex_defects(mesh=mesh)))
        #print(len(v), flush=True)
        #print(np.array(total_area).shape, flush=True)
        #print(v.shape, flush=True)
        #discrete_curv1 = discrete_curv1/(np.array(total_area))
        #print(len(discrete_curv1))
        hist, bins = np.histogram(discrete_curv1, bins=20)
        bins[np.where(bins == 0)] = 1e-15
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.xscale('log')
        plt.hist(discrete_curv1, bins=logbins)
        plt.savefig("hist.jpg")
        plt.close()
        #implicit_curv = np.abs(getImplicitCurvatureforSamples(model, latent, curvature_samples, args))

        wheredisccurv1_1e1 = np.where(discrete_curv1 <= 1e-1)[0]
        wheredisccurv1_1e2 = np.where(discrete_curv1 <= 1e-2)[0]
        wheredisccurv1_1e3 = np.where(discrete_curv1 <= 1e-3)[0]
        wheredisccurv1_1e5 = np.where(discrete_curv1 <= 1e-5)[0]
        wheredisccurv1_1e7 = np.where(discrete_curv1 <= 1e-7)[0]
        wheredisccurv1_1e10 = np.where(discrete_curv1 <= 1e-10)[0]
        wheredisccurv1_1e15 = np.where(discrete_curv1 <= 1e-15)[0]
        print(numsamples, flush=True)
        print("radius = ", r, flush=True)

        disccurv_1e1 = ((1.0 - (len(wheredisccurv1_1e1)/numsamples))) 
        disccurv_1e2 = ((1.0 - (len(wheredisccurv1_1e2)/numsamples))) 
        disccurv_1e3 = ((1.0 - (len(wheredisccurv1_1e3)/numsamples))) 
        disccurv_1e5 = ((1.0 - (len(wheredisccurv1_1e5)/numsamples))) 
        disccurv_1e7 = ((1.0 - (len(wheredisccurv1_1e7)/numsamples))) 
        disccurv_1e10 = ((1.0 - (len(wheredisccurv1_1e10)/numsamples))) 
        disccurv_1e15 = ((1.0 - (len(wheredisccurv1_1e15)/numsamples))) 
        d_disccurv_1e1 = (((len(wheredisccurv1_1e1)/numsamples))) 
        d_disccurv_1e2 = (((len(wheredisccurv1_1e2)/numsamples))) 
        d_disccurv_1e3 = (((len(wheredisccurv1_1e3)/numsamples))) 
        d_disccurv_1e5 = (((len(wheredisccurv1_1e5)/numsamples))) 
        d_disccurv_1e7 = (((len(wheredisccurv1_1e7)/numsamples))) 
        d_disccurv_1e10 = (((len(wheredisccurv1_1e10)/numsamples))) 
        d_disccurv_1e15 = (((len(wheredisccurv1_1e15)/numsamples))) 
        disccurv_median = (np.median(discrete_curv1)) 
        disccurv_mean = (np.mean(discrete_curv1)) 


        print("disccurv_1e1 = ", d_disccurv_1e1)
        print("disccurv_1e2 = ", d_disccurv_1e2)
        print("disccurv_1e3 = ", d_disccurv_1e3)
        print("disccurv_1e5 = ", d_disccurv_1e5)
        print("disccurv_1e7 = ", d_disccurv_1e7)
        print("disccurv_1e10 = ", d_disccurv_1e10)
        print("disccurv_1e15 = ", d_disccurv_1e15)
        print("disccurv_median = ", disccurv_median)
        print("disccurv_avg = ", disccurv_mean)
        print("-------------------------------------------", flush=True)

def getGaussAvgOfSurfacePoints(objfile, numsamples, model_path):
    tester = Tester(model_path, 256, 'cuda') 
    #rotv, simplices = getmcubePoints(tester.grid_samples, tester.model.net)
    #tmesh = trimesh.Trimesh(np.array(rotv), np.array(simplices))
    #tmesh.export('reconstruct.obj')

    #mesh = o3d.io.read_triangle_mesh('reconstruct.obj')
    mesh = o3d.io.read_triangle_mesh(objfile)
    #print(mesh, flush=True)
    numsamples = 60000
    pcl = mesh.sample_points_poisson_disk(number_of_points=int(numsamples))
    samples = np.asarray(pcl.points)*2
    trimesh.Trimesh(vertices = samples).export('recon_dh1e2.ply')
    implicit_curv = getImplicitCurvatureforSamples(tester.model, samples)
    #trisamples,_ = trimesh.sample.sample_surface_even(tmesh, numsamples)

    #implicit_curv = getImplicitCurvatureforSamples(tester.model, trisamples)
    #implicit_curv = getImplicitCurvatureforSamples(tester.model, samples)
    print(implicit_curv.shape, flush=True)
    print("imp curvature = ", implicit_curv)
    print(len(implicit_curv), flush=True)
    print(max(implicit_curv))
    print(min(implicit_curv))

    hist, bins = np.histogram(implicit_curv, bins=20)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.xscale('log')
    plt.hist(implicit_curv, bins=logbins)
    #plt.hist(implicit_curv, bins=20)
    plt.savefig("histimp.jpg") #discrete_curv1 = (np.abs(np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh=mesh, points=curvature_samples, radius=0))))
    
    whereimpcurv_1e1 = np.where(implicit_curv <= 1e-1)[0]
    whereimpcurv_1e3 = np.where(implicit_curv <= 1e-3)[0]
    whereimpcurv_1e5 = np.where(implicit_curv <= 1e-5)[0]
    whereimpcurv_1e2 = np.where(implicit_curv <= 1e-2)[0]

    impcurv_1e1 = (1.0 - (len(whereimpcurv_1e1)/numsamples))
    impcurv_1e3 = (1.0 - (len(whereimpcurv_1e3)/numsamples))
    impcurv_1e5 = (1.0 - (len(whereimpcurv_1e5)/numsamples))
    impcurv_1e2 = (1.0 - (len(whereimpcurv_1e2)/numsamples))

    i_impcurv_1e1 = ((len(whereimpcurv_1e1)/numsamples))
    i_impcurv_1e3 = ((len(whereimpcurv_1e3)/numsamples))
    i_impcurv_1e5 = ((len(whereimpcurv_1e5)/numsamples))
    i_impcurv_1e2 = ((len(whereimpcurv_1e2)/numsamples))

    impcurv_median = np.median(bins)
    impcurv_mean = np.mean(hist)

    print("impcurv_1e1 = ", i_impcurv_1e1)
    print("impcurv_1e2 = ", i_impcurv_1e2)
    print("impcurv_1e3 = ", i_impcurv_1e3)
    print("impcurv_1e5 = ", i_impcurv_1e5)
    print("impcurv_median = ", impcurv_median)
    print("impcurv_avg = ", impcurv_mean)

def getDiscAndImpCurvatureOfSurface(dataset, latent, model, epoch, count, args, prefname="best"):
    samples, surface_area, threshcurv, meancurv, mediancurv = getGaussAvgOfSurfacePoints(dataset, latent, model, epoch, count, args, prefname)

    if len(samples) == 0:
        return [], 0,1000,1000,1000
   
    return samples, surface_area, threshcurv, meancurv, mediancurv

if __name__ == '__main__':

    print("DRAGON-----------")
    #getColorCodedDeterminantforSamples('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impdet_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test')
    #exit()
#    getColorCodedGaussforSamples('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test')
    getColorCodedVarforSamples('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_var_10.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_10_test')
    getColorCodedVarforSamples('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_var_50.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_50_test')
    getColorCodedVarforSamples('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_var_100.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_100_test')
    exit()
#    getColorCodedDeterminantforSamples('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_impdet_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test')
#    getColorCodedGaussforSamples('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test')
#    getColorCodedVarforSamples('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_var.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test')
#    getColorCodedDeterminantforSamples('../../npyfiles/dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_imppoints.npy','../../npyfiles/dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_impdet_hist.npy','dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test')
#    getColorCodedGaussforSamples('../../npyfiles/dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_imppoints.npy','../../npyfiles/dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test')
#    getColorCodedVarforSamples('../../npyfiles/dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_imppoints.npy','../../npyfiles/dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_var.npy','dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test')
#    exit()
#
#    print("GRIFFIN-----------")
#    getColorCodedDeterminantforSamples('../../npyfiles/griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impdet_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test')
#    getColorCodedGaussforSamples('../../npyfiles/griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impgauss_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test')
#    getColorCodedVarforSamples('../../npyfiles/griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_var.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test')
#    getColorCodedDeterminantforSamples('../../npyfiles/griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_imppoints.npy','../../npyfiles/griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_impdet_hist.npy','griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test')
#    getColorCodedGaussforSamples('../../npyfiles/griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_imppoints.npy','../../npyfiles/griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test')
#    getColorCodedVarforSamples('../../npyfiles/griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_imppoints.npy','../../npyfiles/griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_var.npy','griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test')
#    exit()
#    curvatureHistogram('../../griffin_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','griffin_gelu_lr1e4_dn0_notanh_test_impgauss', 20)
#    curvatureHistogram('../../griffin_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','griffin_gelu_lr1e4_dn0_notanh_test_impmin',5)
#    curvatureHistogram('../../griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_impgauss', 20)
#    curvatureHistogram('../../griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_impmin',5)
#
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_impgauss_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_impgauss',20)
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_impmin_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_impmin',5)
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_test_impgauss_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_test_impgauss',20)
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_test_impmin_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_test_impmin',5)
#
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impgauss_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impgauss',20)
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impmin_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impmin',5)
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_impgauss_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_impgauss',20)
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_impmin_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_impmin',5)
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_impgauss_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_impgauss',20)
#    curvatureHistogram('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_impmin_hist.npy','griffin_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_impmin',5)
#    npyfile = '../../data/input/500k_sampled/griffin.npy'
#    numsamples = 500000
#    getCham('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../griffin_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_512.obj', '../../data/input/obj/griffin_gaussthin_2_25.obj', numsamples)
#    getCham('../../griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../griffin_gelu_lr1e4_dn0_notanh_test_512.obj', npyfile,numsamples)
#    exit()

#    print("DRAGON-----------")
#    curvatureHistogram('../../dragon_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_dn0_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_dn0_notanh_impmin_test',5)
#
#    curvatureHistogram('../../dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_impmin_test',5)
#
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_impmin_test',5)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_impmin_test',5)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3e1_svd3_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh3e1_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3e1_svd3_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh3e1_svd3_notanh_impmin_test',5)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_impmin_test',5)
#
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_impmin_test',5)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_impmin_test',5)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_impmin_test',5)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_impmin_test',5)
#
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_impmin_test',5)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_impgauss_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_impgauss_test',20)
#    curvatureHistogram('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_impmin_hist.npy','dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_impmin_test',5)

#    npyfile = '../../data/input/500k_sampled/dragon.npy'
#    numsamples = 500000
#    getCham('../../dragon_gelu_lr1e4_dn0_notanh_test_512.obj',npyfile,numsamples)
#    getCham('../../dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_512.obj', 
#            '../../data/input/obj/dragon_25k_gaussthin_100_7_60.obj', numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh2e1_svd3_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3e1_svd3_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_512.obj', npyfile,numsamples)
#    exit()

#    print("HORSE-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_dn0_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_dn0_notanh_impmin_test',5)
#    print("GAUSSTHIN-----------")
#    curvatureHistogram('../../horse_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','horse_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','horse_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_impmin_test',5)
#    print("ODED-----------")
#    curvatureHistogram('../../horse_oded_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','horse_oded_gelu_lr1e4_dn0_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_oded_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','horse_oded_gelu_lr1e4_dn0_notanh_impmin_test',5)
#    print("SVD-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_impmin_test',5)
#    print("LOGDET 3-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_impmin_test',5)
#    print("HESS 1-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_impmin_test',5)
#    print("HESS 3-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_impmin_test',5)
#    print("HESS 5-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_impmin_test',5)
#    print("SVD3 5-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5_svd3_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh5_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5_svd3_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh5_svd3_notanh_impmin_test',5)
#    print("SVD3 1e1-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_impmin_test',5)
#    print("SVD3 5e1-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_impmin_test',5)
#    print("SVD3 1e2-----------")
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_test_impgauss_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_test_impmin_hist.npy','horse_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_impmin_test',5)
#    npyfile = '../../data/input/250k_sampled/horse.npy'
#    numsamples=250000
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_test_512.obj', npyfile,numsamples)
#    exit()
#    print("HORSE-----------")
#    npyfile = '../../data/input/250k_sampled/horse.npy'
#    numsamples=250000
#    getCham('../../horse_gelu_lr1e4_dn0_notanh_test_512.obj',npyfile,numsamples)
#    print("ODED-----------")
#    getCham('../../horse_oded_gelu_lr1e4_dn0_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../horse_oded_gelu_lr1e4_dn0_notanh_test_512.obj', '../../data/input/obj/horse_oded.obj', numsamples)
#    print("GAUSSTHIN-----------")
#    getCham('../../horse_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../horse_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_512.obj', '../../data/input/obj/horse_gaussthin_100_7_60.obj', numsamples)
#    print("SVD-----------")
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1_svd_notanh_test_512.obj', npyfile,numsamples)
#    print("LOGDET-----------")
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh3_logdet_notanh_test_512.obj', npyfile,numsamples)
#    print("HESSIAN 1-----------")
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    print("HESSIAN 3-----------")
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh3_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    print("HESSIAN 5-----------")
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    print("SVD3 5-----------")
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5_svd3_notanh_test_512.obj', npyfile,numsamples)
#    print("SVD3 1e1-----------")
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh_test_512.obj', npyfile,numsamples)
#    print("SVD3 5e1-----------")
#    getCham('../../horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh_test_512.obj', npyfile,numsamples)
#    exit()


#    print("BUNNY---")
    #getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_128.obj', 250000, scale=1, radius=[0]) 
    #getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_512.obj', 250000, scale=1, radius=[0]) 
    #getDiscreteCurvatureSurfacePoints('../../stanford_bunny128_gaussthin_2_25.obj', 250000, scale=1, radius=[0]) 
#    getDiscreteCurvatureSurfacePoints('../../stanford_bunny256_gaussthin_2_25.obj', 250000, scale=1, radius=[0]) 
    #getDiscreteCurvatureSurfacePoints('../../data/input/obj/bunny_high.obj', 250000, scale=1, radius=[0]) 
#    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_128.obj', 250000, scale=1, radius=[0])
#    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_512.obj', 250000, scale=1, radius=[0])
#    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_1024.obj', 250000, scale=1, radius=[0])
    #getCham('../../stanford_bunny128_gaussthin_2_25.obj', '../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_128.obj')
#    getCham('../../stanford_bunny256_gaussthin_2_25.obj', '../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_256.obj')
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_128.obj', '../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_128.obj')
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_512.obj', '../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_512.obj')
    #getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_1024.obj', '../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_1024.obj')
#    exit()
    #curvatureHistogram('../../bunny_oded1_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','bunny_oded1_gelu_lr1e4_dn0_notanh_impgauss_test')
    #curvatureHistogram('../../bunny_oded1_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','bunny_oded1_gelu_lr1e4_dn0_notanh_impmin_test')
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_hessiandethat_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    getCham('../../output/bunny_oded1_gelu_lr1e4_dn0_notanh_nodrop_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    exit()
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','stanford_bunny_gelu_lr1e4_dn0_notanh_impgauss_test',20)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','stanford_bunny_gelu_lr1e4_dn0_notanh_impmin_test',5)
#    print("BUNNY HIGH ---")
#    curvatureHistogram('../../bunny_high_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','bunny_high_gelu_lr1e4_dn0_notanh_impgauss_test',20)
#    curvatureHistogram('../../bunny_high_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','bunny_high_gelu_lr1e4_dn0_notanh_impmin_test',5)
#    print("BUNNY ODED ---")
#    curvatureHistogram('../../bunny_oded1_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','bunny_oded1_gelu_lr1e4_dn0_notanh_impgauss_test',20)
#    curvatureHistogram('../../bunny_oded1_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','bunny_oded1_gelu_lr1e4_dn0_notanh_impmin_test',5)
#    print("SVD ---")
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_test_impgauss_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_impgauss_test',20)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_test_impmin_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_impmin_test',5)
#    print("LOGDET ---")
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_test_impgauss_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_impgauss_test',20)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_test_impmin_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_impmin_test',5)
#    print("SVD3 2---")
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_test_impgauss_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_impgauss_test',20)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_test_impmin_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_impmin_test',5)
#    #curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_svd3_notanh_test_impgauss_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_svd3_notanh_impgauss_test',20)
#    #curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_svd3_notanh_test_impmin_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_svd3_notanh_impmin_test',5)
#    print("HESSDET 2 ---")
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_impgauss_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_impmin_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_impmin_test',5)
#    print("hessdet 3 ---")
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_hessiandethat_notanh_test_impgauss_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_hessiandethat_notanh_impgauss_test',20)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_hessiandethat_notanh_test_impmin_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e3_hessiandethat_notanh_impmin_test',5)
#    exit()
    #curvatureHistogram('../../bunny_oded_gelu_lr1e4_dn0_notanh_test_impgauss_hist.npy','bunny_oded_gelu_lr1e4_dn0_notanh_impgauss_test',20)
    #curvatureHistogram('../../bunny_oded_gelu_lr1e4_dn0_notanh_test_impmin_hist.npy','bunny_oded_gelu_lr1e4_dn0_notanh_impmin_test',5)
#    npyfile = '../../data/input/250k_sampled/stanford_bunny.npy'
#    numsamples=250000
    #getCham('../../stanford_bunny_gelu_lr1e4_dn0_notanh_test_512.obj',npyfile,numsamples)
    #print("ODED-----------")
#    getCham('../../bunny_oded1_gelu_lr1e4_dn0_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../bunny_oded1_gelu_lr1e4_dn0_notanh_test_512.obj', '../../data/input/obj/bunny_oded1.obj', numsamples)
#    exit()
#    print("GAUSSTHIN-----------")
#    getCham('../../bunny_high_gelu_lr1e4_dn0_notanh_test_512.obj', npyfile,numsamples)
#    getCham('../../bunny_high_gelu_lr1e4_dn0_notanh_test_512.obj', '../../data/input/obj/bunny_high.obj', numsamples)
#    print("SVD-----------")
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_test_512.obj', npyfile,numsamples)
#    print("LOGDET-----------")
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_test_512.obj', npyfile,numsamples)
#    print("HESSIAN -----------")
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_512.obj', npyfile,numsamples)
#    print("SVD3 5-----------")
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_test_512.obj', npyfile,numsamples)
#    exit()
    print("LUCY----------")
    #r = [0, 0.01, 0.02, 0.05]
    getColorCodedDeterminantforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test_impdet_hist.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test')
    getColorCodedGaussforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test_impgauss_hist.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test')
    getColorCodedVarforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test_var.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test')
    getColorCodedDeterminantforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test_impdet_hist.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test')
    getColorCodedGaussforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test_impgauss_hist.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test')
    getColorCodedVarforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test_var.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test')
    getColorCodedDeterminantforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_impdet_hist.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test')
    getColorCodedGaussforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_impgauss_hist.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test')
    getColorCodedVarforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_var.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test')
    getColorCodedDeterminantforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impdet_hist.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test')
    getColorCodedGaussforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impgauss_hist.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test')
    getColorCodedVarforSamples('../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_var.npy','lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test')
    exit()
    #r = [0.05]
    #getDiscreteCurvatureSurfacePoints('../../lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_test_1024.obj', 250000, scale=1, radius=r)
    #r = [0, 0.01, 0.02, 0.05]
    #getDiscreteCurvatureSurfacePoints('../../obj/lucy/lucy_gaussthin_300_7_5.obj', 250000, scale=1, radius=r)
    #exit()

    curvatureHistogram('../../output/bunny_oded1_gelu_lr1e4_dn0_notanh_nodrop_imp_hist.npy','bunny_oded1_gelu_lr1e4_dn0_notanh')
    curvatureHistogram('../../bunny_high_gelu_lr1e4_dn0_notanh_resume_imp_hist.npy','bunny_high_gelu_lr1e4_dn0_notanh')
    curvatureHistogram('../../output/stanford_bunny_gelu_lr1e4_dn0_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_dn0_notanh')
    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh')
    print("EINSTEIN-----------")
    curvatureHistogram('../../output/einstein_gelu_lr1e4_dn0_notanh_imp_hist.npy','einstein_gelu_lr1e4_dn0_notanh')
    curvatureHistogram('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_imp_hist.npy','einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh')
    curvatureHistogram('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_imp_hist.npy','einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh')
    curvatureHistogram('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_imp_hist.npy','einstein_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh')
    curvatureHistogram('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_imp_hist.npy','einstein_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh')
    curvatureHistogram('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_imp_hist.npy','einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh')
    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh')
    exit()

    getCham('../../output/einstein_gelu_lr1e4_dn0_notanh_512.obj','../../data/input/250k_sampled/einstein.npy')
    getCham('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_512.obj', '../../data/input/250k_sampled/einstein.npy')
    getCham('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_512.obj', '../../data/input/250k_sampled/einstein.npy') 
    getCham('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_512.obj', '../../data/input/250k_sampled/einstein.npy') 
    getCham('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_512.obj','../../data/input/250k_sampled/einstein.npy') 
    getCham('../../einstein_gelu_lr1e4_hlr1e-5_dn0_dh1e-1_hessiandethat_notanh_512.obj', '../../data/input/250k_sampled/einstein.npy')
    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    exit()

    r = [0, 0.01, 0.02, 0.05]
    getCham('../../bunny_high_gelu_lr1e4_dn0_notanh_resume_512.obj', '../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj') #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    getCham('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    getCham('../../bunny_high_gelu_lr1e4_dn0_notanh_resume_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    getDiscreteCurvatureSurfacePoints('../../bunny_high_gelu_lr1e4_dn0_notanh_resume_512.obj', 250000, scale=1, radius=[0]) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_512.obj', 250000, scale=1, radius=[0])
    exit()
    #print("oded stein")
    #print("---------------------\n")
    #getDiscreteCurvatureSurfacePoints('../../bunny_oded1.obj', 250000, scale=1, radius=r) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getCham('../../bunny_oded1.obj', '../../data/input/250k_sampled/stanford_bunny.npy')

    #print("bunny high")
    #print("---------------------\n")
    #getDiscreteCurvatureSurfacePoints('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', 250000, scale=1, radius=r) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getCham('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', '../../data/input/250k_sampled/stanford_bunny.npy')


    print("svd3")
    print("---------------------\n")
    #getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_512.obj', 250000, scale=1, radius=r)
    #getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_128.obj', 250000, scale=1, radius=r)
    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_128.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    exit()


    print("hessiandethat")
    print("---------------------\n")
    #getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_512.obj', 250000, scale=1, radius=r)
    #getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_1024.obj', 250000, scale=1, radius=r)
    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_1024.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    exit()


    print("logdet")
    print("---------------------\n")
    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_512.obj', 250000, scale=1, radius=r)
    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')


    print("SVD")
    print("---------------------\n")
    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_512.obj', 250000, scale=1, radius=r)
    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')

    exit()
#    print("sdf", flush=True)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_dn0_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_dn0_notanh')
#    getCham('../../output/stanford_bunny_gelu_lr1e4_dn0_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    print("bunny_high", flush=True)
#    curvatureHistogram('../../bunny_high_gelu_lr1e4_dn0_notanh_resume_imp_hist.npy', 'bunny_high_gelu_lr1e4_dn0_notanh_imp_hist')
    getCham('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    print("hessdethat", flush=True)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh')
    #getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    #getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_1024.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_test_1024.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    print("hessdet", flush=True)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandet_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandet_notanh')
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandet_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    print("logdet", flush=True)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh')
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_logdet_notanh_256.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    print("svd", flush=True)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh')
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd_notanh_256.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    print("svd3", flush=True)
#    curvatureHistogram('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_imp_hist.npy','stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh')
#    getCham('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_512.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
#    exit()
    #getGaussAvgOfSurfacePoints('../../bunny_high_gelu_lr1e4_dn0_notanh_train.obj', 250000, '../../checkpoints/bunny_high_gelu_lr1e4_dn0_notanh/best_train_loss.tar')
    #getGaussAvgOfSurfacePoints('../../bunny_silu_hessdet_dh1e2_gelu_lr1e4_dn0_notanh_train.obj', 250000, '../../checkpoints/bunny_silu_hessdet_dh1e2_gelu_lr1e4_dn0_notanh/best_train_loss.tar')
    #getGaussAvgOfSurfacePoints('../../lucy_gelu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_notanh_reg1_grid128_train.obj', 250000, '../../checkpoints/lucy_gelu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_notanh_reg1_grid128/best_train_loss.tar')
    #getGaussAvgOfSurfacePoints('../../250k_sampled/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1_train.obj', 250000)
    #getDiscreteCurvatureSurfacePoints('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', 60000, scale=1, radius=0)
    #getDiscreteCurvatureSurfacePoints('../../lucy_gelu_lr1e4_hlr1e5_dn0_dh1e3_hessiandet_notanh_reg1_grid128_train.obj', 60000, scale=1, radius=0) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getDiscreteCurvatureSurfacePoints('../../250k_sampled/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1_train.obj', 60000, scale=1, radius=0) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getDiscreteCurvatureSurfacePoints('../../bunny_high_gelu_lr1e4_dn0_notanh_train.obj', 60000, radius=0) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_512.obj', 60000, scale=1, radius=0) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_test_1024.obj', 60000, scale=1, radius=0) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_svd3_notanh_256.obj', 60000, scale =1, radius=0.05) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getDiscreteCurvatureSurfacePoints('../../stanford_bunny_gelu_lr1e4_hlr1e-5_dn0_dh1e2_hessiandethat_notanh_512.obj', 60000, scale =1, radius=0)
    getDiscreteCurvatureSurfacePoints('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', 60000, radius=0) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getDiscreteCurvatureSurfacePoints('../../bunny_high_gelu_lr1e4_dn0_notanh_resume_256.obj', 60000, radius=0) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getDiscreteCurvatureSurfacePoints('../../bunny_silu_hessdet_dh1e2_gelu_lr1e4_dn0_notanh_train.obj', 60000, radius=0) #, '../../checkpoints/silu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_withtanh_reg1/best_train_loss.tar')
    #getGaussAvgOfSurfacePoints('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', 250000)
    #getCham('../../lucy_gelu_lr1e4_hlr1e5_dn0_dh1e2_svd3_notanh_reg1_grid128_train.obj', '../../data/input/250k_sampled/stanford_bunny.npy')

    #getCham1('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', '../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/stanford_bunny_input.obj')
    #getHauss('../../lucy_gelu_lr1e4_hlr1e5_dn0_dh1e2_hessiandet_notanh_reg1_grid128_train.obj', '../../data/input/250k_sampled/stanford_bunny.npy')
    #getHauss1('../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/bunny_high.obj', '../../../DevelopableApproximationViaGaussImageThinning/examples/bunny_high/stanford_bunny_input.obj')
