import os
from dataset import gridData
from loadmodel import loadEvaluateModel, loadPretrainedModel
from trainhelper import getDiscAndImpCurvatureOfSurface, getColorCurvatureOfSurface, getDiscreteCurvatureOfSurface
from inference import latCodeOptimization
from initialize import getPointNormal
from utils import normalize_pts_withdia, convertToPLY, getChamferDist
import torch
import json
import numpy as np
import open3d as o3d
import trimesh
from trainhelper import getSurfaceSamplePoints 


deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)

class reconstructDict():
    def __init__(self):
        self.dict = {'imp':{'thresh1e-1':{}, 'thresh1e-2':{}, 'thresh1e-3':{}, 'mean':{}, 'median':{}}, 'disc':{'thresh1e-1':{}, 'thresh1e-2':{}, 'thresh1e-3':{}, 'mean':{}, 'median':{}}, 'cham':{}}

def reconstruct(model, args):
    grid_uniformsamples = gridData(args=args)
    #for use_model in ['combmodellatloss', 'combhesslatloss', 'modellatloss',  'hesslatloss', 'lat']:       
    #for use_model in ['combmodellat', 'combhesslat', 'modellat',  'hesslat', 'lat']:       
    #model, _,_, _ = loadPretrainedModel(model, args)
    #getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 300000, args, 'lat')
    #samples = latCodeOptimization(args)
    #chamfer(samples, args)
    alltestfile = open(args.testfilepath,'r')
    for fname in alltestfile:
        fname = fname.strip()
        basefname = fname[12:]
        #for use_model in ['lat', 'reg0', 'svd_reg1', 'svd3_reg10.0', 'logdet_reg1']: # 'hesslat',  'combmodellat', 'combhesslat']:       
        for use_model in ['reg0', 'svd3_reg10.0']: # 'hesslat',  'combmodellat', 'combhesslat']:       
            if use_model in ['lat']:
                if not os.path.exists(os.path.join(args.checkpoint_folder, 'model_best_'+use_model+'_'+fname+'.pth.tar')):
                    print(use_model+" does not exists")
                    continue
                model, latent = loadEvaluateModel(model, use_model+'_'+fname, args)
            else:
                #if not os.path.exists(os.path.join(args.checkpoint_folder, 'model_best_'+use_model+'_06_05_'+fname+'.pth.tar')):
                if not os.path.exists(os.path.join(args.checkpoint_folder, 'model_best_'+use_model+'_'+fname+'.pth.tar')):
                    print(use_model+" does not exists")
                    continue
                model, latent = loadEvaluateModel(model, use_model+'_'+fname, args)
                if model is None:
                    return
                model.to(device)
            print("loaded evaluation {} model".format(use_model))
            #samples, _, threshcurv, meancurv, mediancurv = getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 300000, args, 'reconstruct_'+use_model+'_06_05_'+fname)
            #if len(samples) <= 0:
            #    print(fname +" "+use_model+" IF level is 0.. rerun")
            #    continue
            #chamdist = chamfer(samples, basefname, args)
            #samples, _, _, _, _= getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 300000, args, use_model)
            getColorCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 300000, args, use_model+'_'+fname)
            #chamfer(samples, args)
    return

def getcurvatureandchamfer(model, modeltype):
        model, latent = loadEvaluateModel(model, modeltype, args)
        if model is None:
            return
        model.to(device)
        print("loaded evaluation {} model".format(use_model))
        samples, _, _, _, _= getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 300000, args, modeltype)
        cham_dist = chamfer(samples, args)
        return 

def reconstructTrain(args):
    grid_uniformsamples = gridData(args=args)
    basemodel, baselatent ,_, _ = loadPretrainedModel(None, args)
    basemodel.to(device)
    samples = getSurfaceSamplePoints(grid_uniformsamples, baselatent(torch.tensor(0)), basemodel, -1, 300000, args, 'best0')
    samples = getSurfaceSamplePoints(grid_uniformsamples, baselatent(torch.tensor(10)), basemodel, -1, 300000, args, 'best10')
    samples = getSurfaceSamplePoints(grid_uniformsamples, baselatent(torch.tensor(20)), basemodel, -1, 300000, args, 'best20')

def poissonSurfaceReconstruction(args):
    alltestfile = open(args.testfilepath,'r')
    #test_dataset = initDeepsdfTestDataSet(args, filename)
    #indexfname_dict = test_dataset.getindexfname()
    lat = reconstructDict()
    #for index,fname in indexfname_dict.items():
    for fname in alltestfile:
        fname = fname.strip()
        if "noisy_0.01" in fname:
            basefname = fname[11:]
        npy = np.load(os.path.join(args.testdir, fname+'.npy'))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(npy[:,0:3])
        pcd.normals = o3d.utility.Vector3dVector(npy[:,3:])
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)
        o3d.io.write_triangle_mesh(fname+'.obj', mesh)
        #mesh = trimesh.exchange.obj.load_obj(fname+'.obj', force='mesh')
        samples, _, threshcurv, meancurv, mediancurv = getDiscreteCurvatureOfSurface(np.asarray(mesh.vertices), np.asarray(mesh.triangles), args, 'reconstruct_lat_'+fname)
        chamdist = chamfer(samples, basefname, args)
        lat.dict['disc']['thresh1e-1'][fname] = threshcurv[0][0]
        lat.dict['disc']['thresh1e-2'][fname] = threshcurv[0][1]
        lat.dict['disc']['thresh1e-3'][fname] = threshcurv[0][2]
        lat.dict['disc']['mean'][fname] = meancurv[0]
        lat.dict['disc']['median'][fname] = mediancurv[0]
        lat.dict['cham'][fname] = chamdist
    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat.json'),'w') as fwrite:
        json.dump(lat.dict, fwrite)


def reconstructallForChamfer(args):
    grid_uniformsamples = gridData(args=args)
    basemodel, _,_, _ = loadPretrainedModel(None, args)
    basemodel.to(device)

    alltestfile = open(args.testfilepath,'r')
    lat, reg0, svd3_reg1, svd_reg1, logdetT_reg1, eikonal_reg1 = reconstructDict() ,reconstructDict(),reconstructDict(),reconstructDict(),reconstructDict(),reconstructDict()
    svd3_reg10, svd_reg10, logdetT_reg10, dataeikonal_reg10, eikonal_reg10 = reconstructDict(),reconstructDict(),reconstructDict(),reconstructDict(), reconstructDict()

    index = 0
    def loadDictValues(dictionary, fname, chamdist):
        dictionary.dict['cham'][fname] = chamdist
        return dictionary

    for fname in alltestfile:
        fname = fname.strip()
        basefname = fname[11:]

        #for use_model in ['lat','reg0.0', 'reg10.0']: # 'hesslat',  'combmodellat', 'combhesslat']:       
        #for use_model in ['lat', 'reg0','svd3_reg10.0_06_05', 'svd_reg1_06_05', 'logdet_reg1_06_05']: # 'hesslat',  'combmodellat', 'combhesslat']:       
        #for use_model in ['lat', 'reg0','svd3_reg10.0_06_05', 'svd_reg1_06_05', 'logdetT_reg1_06_05', 'eikonal_reg1_06_05']: 
        for use_model in ['svd3_reg10.0_06_05', 'logdetT_reg1_06_05', 'reg10.0_dataeikonal', 'svd_reg1_06_05', 'reg0', 'lat']: 
        #for use_model in ['reg10.0_dataeikonal']: 
        #for use_model in ['lat']: 
        #for use_model in ['svd3_reg10.0_06_05']: 
            if use_model == 'lat':
                print("latent only\n")
                latpath = (os.path.join(args.latcode_folder, 'test_latent_'+args.latfname+'_'+fname+'.py'))
                if not os.path.exists(latpath):
                    print(fname +"lat does not exists - re-run")
                    continue
                latent = torch.load(os.path.join(args.latcode_folder, 'test_latent_'+args.latfname+'_'+fname+'.py'))
                samples = getSurfaceSamplePoints(grid_uniformsamples, latent, basemodel, -1, 250000, args, 'reconstruct_'+use_model+'_'+fname)
                chamdist = chamfer(samples, basefname, args)
                #lat = loadDictValues(lat, fname, threshcurv, meancurv, mediancurv, chamdist)
                lat = loadDictValues(lat, fname, chamdist)
            else:
                model, latent  = loadEvaluateModel(None, use_model+'_'+fname, args, args.checkpoint_folder)
                if model is None:
                    print(fname +"does not exists - re-run")
                    continue
                model.to(device)
                print("loaded evaluation {} model".format(use_model))
                samples = getSurfaceSamplePoints(grid_uniformsamples, latent, basemodel, -1, 250000, args, 'reconstruct_'+use_model+'_'+fname)
                if len(samples) <= 0:
                    print(fname +" "+use_model+" IF level is 0.. rerun")
                    continue
                chamdist = chamfer(samples, basefname, args)

                if use_model == 'reg0':
                    reg0 = loadDictValues(reg0, fname, chamdist)
                elif use_model == 'lat':
                    lat = loadDictValues(lat, fname, chamdist)
                elif use_model == 'svd3_reg1_06_05':
                    svd3_reg1 = loadDictValues(svd3_reg1, fname, chamdist)
                elif use_model == 'svd_reg1_06_05':
                    svd_reg1 = loadDictValues(svd_reg1, fname, chamdist)
                elif use_model == 'logdetT_reg1_06_05':
                    logdetT_reg1 = loadDictValues(logdetT_reg1, fname, chamdist)
                elif use_model == 'eikonal_reg1_06_05':
                    eikonal_reg1 = loadDictValues(eikonal_reg1, fname, chamdist)
                elif use_model == 'svd3_reg10.0_06_05':
                    svd3_reg10 = loadDictValues(svd3_reg10, fname, chamdist)
                elif use_model == 'svd_reg10.0_06_05':
                    svd_reg10 = loadDictValues(svd_reg10, fname, chamdist)
                elif use_model == 'logdetT_reg10.0_06_05':
                    logdetT_reg10 = loadDictValues(logdetT_reg10, fname, chamdist)
                elif use_model == 'eikonal_reg10.0_06_05':
                    eikonal_reg10 = loadDictValues(eikonal_reg10, fname, chamdist)
                elif use_model == 'reg10.0_dataeikonal':
                    dataeikonal_reg10 = loadDictValues(dataeikonal_reg10, fname, chamdist)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat.json'),'w') as fwrite:
        json.dump(lat.dict, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0.json'),'w') as fwrite:
        json.dump(reg0.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1.json'),'w') as fwrite:
#        json.dump(svd3_reg1.dict, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1.json'),'w') as fwrite:
        json.dump(svd_reg1.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_eikonal_reg1.json'),'w') as fwrite:
#        json.dump(eikonal.dict, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_dataeikonal_reg10.json'),'w') as fwrite:
        json.dump(dataeikonal_reg10.dict, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdetT_reg1.json'),'w') as fwrite:
        json.dump(logdetT_reg1.dict, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.json'),'w') as fwrite:
        json.dump(svd3_reg10.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg10.json'),'w') as fwrite:
#        json.dump(svd.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_eikonal_reg10.json'),'w') as fwrite:
#        json.dump(eikonal_reg10.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdetT_reg10.json'),'w') as fwrite:
#        json.dump(logdetT_reg10.dict, fwrite)

def reconstructall(args):
    if args.losstype == 'poisson':
        poissonSurfaceReconstruction(args)

    grid_uniformsamples = gridData(args=args)
    basemodel, _,_, _ = loadPretrainedModel(None, args)
    basemodel.to(device)

    alltestfile = open(args.testfilepath,'r')
    lat, reg0, svd3_reg1, svd_reg1, logdetT_reg1, eikonal_reg1 = reconstructDict() ,reconstructDict(),reconstructDict(),reconstructDict(),reconstructDict(),reconstructDict()
    svd3_reg10, svd_reg10, logdetT_reg10, dataeikonal_reg10, eikonal_reg10 = reconstructDict(),reconstructDict(),reconstructDict(),reconstructDict(), reconstructDict()

    index = 0
    def loadDictValues(dictionary, fname, threshcurv, meancurv, mediancurv, chamdist):
        dictionary.dict['imp']['thresh1e-1'][fname] = threshcurv[1][0]
        dictionary.dict['imp']['thresh1e-2'][fname] = threshcurv[1][1]
        dictionary.dict['imp']['thresh1e-3'][fname] = threshcurv[1][2]
        dictionary.dict['imp']['mean'][fname] = meancurv[1]
        dictionary.dict['imp']['median'][fname] = mediancurv[1]
        dictionary.dict['disc']['thresh1e-1'][fname] = threshcurv[0][0]
        dictionary.dict['disc']['thresh1e-2'][fname] = threshcurv[0][1]
        dictionary.dict['disc']['thresh1e-3'][fname] = threshcurv[0][2]
        dictionary.dict['disc']['mean'][fname] = meancurv[0]
        dictionary.dict['disc']['median'][fname] = mediancurv[0]
        dictionary.dict['cham'][fname] = chamdist
        return dictionary

    for fname in alltestfile:
        fname = fname.strip()
        if "noisy_0.01" in fname:
            basefname = fname[11:]
        elif "noisy_0.02" in fname:
            basefname = fname[11:]
        elif "noisy_0.005" in fname:
            basefname = fname[12:]
        else:
            basefname = fname

        #for use_model in ['lat','reg0.0', 'reg10.0']: # 'hesslat',  'combmodellat', 'combhesslat']:       
        #for use_model in ['lat', 'reg0','svd3_reg10.0_06_05', 'svd_reg1_06_05', 'logdet_reg1_06_05']: # 'hesslat',  'combmodellat', 'combhesslat']:       
        #for use_model in ['lat', 'reg0','svd3_reg10.0_06_05', 'svd_reg1_06_05', 'logdetT_reg1_06_05', 'eikonal_reg1_06_05']: 
        #for use_model in ['svd3_reg10.0_06_05', 'logdetT_reg1_06_05', 'reg10.0_dataeikonal', 'svd_reg1_06_05', 'reg0', 'lat']: 
        #for use_model in ['svd3_reg10.0_06_05', 'lat']: 
        #for use_model in ['lat', 'reg0', 'svd3_reg10.0_06_05']: 
        for use_model in ['lat']: 
        #for use_model in ['reg10.0_dataeikonal']: 
        #for use_model in ['lat']: 
        #for use_model in ['svd3_reg10.0_06_05']: 
#            if use_model == 'lat':
#                print("latent only\n")
#                latpath = (os.path.join(args.latcode_folder, 'test_latent_'+args.latfname+'_'+fname+'.py'))
#                if not os.path.exists(latpath):
#                    print(fname +"lat does not exists - re-run")
#                    continue
#                latent = torch.load(os.path.join(args.latcode_folder, 'test_latent_'+args.latfname+'_'+fname+'.py'))
#                samples,_, threshcurv, meancurv, mediancurv = getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, basemodel, -1, 300000, args, 'reconstruct_lat_'+fname)
#                chamdist = chamfer(samples, basefname, args)
#                lat = loadDictValues(lat, fname, threshcurv, meancurv, mediancurv, chamdist)
#            else:
                model, latent  = loadEvaluateModel(None, use_model+'_'+fname, args, args.checkpoint_folder)
                if model is None:
                    print(fname +"does not exists - re-run")
                    continue
                model.to(device)
                print("loaded evaluation {} model".format(use_model))
                samples, _, threshcurv, meancurv, mediancurv = getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 250000, args, 'reconstruct_'+use_model+'_'+fname)
                if len(samples) <= 0:
                    print(fname +" "+use_model+" IF level is 0.. rerun")
                    continue
                chamdist = chamfer(samples, basefname, args)

                if use_model == 'reg0':
                    reg0 = loadDictValues(reg0, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'lat':
                    lat = loadDictValues(lat, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'svd3_reg1_06_05':
                    svd3_reg1 = loadDictValues(svd3_reg1, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'svd_reg1_06_05':
                    svd_reg1 = loadDictValues(svd_reg1, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'logdetT_reg1_06_05':
                    logdetT_reg1 = loadDictValues(logdetT_reg1, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'eikonal_reg1_06_05':
                    eikonal_reg1 = loadDictValues(eikonal_reg1, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'svd3_reg10.0_06_05':
                    svd3_reg10 = loadDictValues(svd3_reg10, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'svd_reg10.0_06_05':
                    svd_reg10 = loadDictValues(svd_reg10, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'logdetT_reg10.0_06_05':
                    logdetT_reg10 = loadDictValues(logdetT_reg10, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'eikonal_reg10.0_06_05':
                    eikonal_reg10 = loadDictValues(eikonal_reg10, fname, threshcurv, meancurv, mediancurv, chamdist)
                elif use_model == 'reg10.0_dataeikonal':
                    dataeikonal_reg10 = loadDictValues(dataeikonal_reg10, fname, threshcurv, meancurv, mediancurv, chamdist)

    
    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat.json'),'w') as fwrite:
        json.dump(lat.dict, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0.json'),'w') as fwrite:
        json.dump(reg0.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1.json'),'w') as fwrite:
#        json.dump(svd3_reg1.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1.json'),'w') as fwrite:
#        json.dump(svd_reg1.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_eikonal_reg1.json'),'w') as fwrite:
#        json.dump(eikonal.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_dataeikonal_reg10.json'),'w') as fwrite:
#        json.dump(dataeikonal_reg10.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdetT_reg1.json'),'w') as fwrite:
#        json.dump(logdetT_reg1.dict, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.json'),'w') as fwrite:
        json.dump(svd3_reg10.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg10.json'),'w') as fwrite:
#        json.dump(svd.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_eikonal_reg10.json'),'w') as fwrite:
#        json.dump(eikonal_reg10.dict, fwrite)
#    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdetT_reg10.json'),'w') as fwrite:
#        json.dump(logdetT_reg10.dict, fwrite)

def reconstructall_bkup(args):
    grid_uniformsamples = gridData(args=args)
    basemodel, _,_, _ = loadPretrainedModel(None, args)
    basemodel.to(device)

    alltestfile = open(args.testfilepath,'r')
    lat_impthresh, reg0_impthresh, svd3_reg1e1_impthresh, svd_reg1_impthresh, logdet_reg1_impthresh = {}, {}, {}, {}, {}
    lat_impmeancurv, reg0_impmeancurv, svd3_reg1e1_impmeancurv, svd_reg1_impmeancurv, logdet_reg1_impmeancurv = {}, {}, {}, {}, {}
    lat_impmediancurv, reg0_impmediancurv, svd3_reg1e1_impmediancurv, svd_reg1_impmediancurv, logdet_reg1_impmediancurv = {}, {}, {}, {}, {}
    lat_discthresh, reg0_discthresh, svd3_reg1e1_discthresh, svd_reg1_discthresh, logdet_reg1_discthresh = {}, {}, {}, {}, {}
    lat_discmeancurv, reg0_discmeancurv, svd3_reg1e1_discmeancurv, svd_reg1_discmeancurv, logdet_reg1_discmeancurv = {}, {}, {}, {}, {}
    lat_discmediancurv, reg0_discmediancurv, svd3_reg1e1_discmediancurv, svd_reg1_discmediancurv, logdet_reg1_discmediancurv = {}, {}, {}, {}, {}
    lat_chamdist, reg0_chamdist, svd3_reg1e1_chamdist, svd_reg1_chamdist, logdet_reg1_chamdist = {}, {}, {}, {}, {}
    index = 0
    for fname in alltestfile:
        fname = fname.strip()
        basefname = fname[11:]

        #for use_model in ['lat','reg0.0', 'reg10.0']: # 'hesslat',  'combmodellat', 'combhesslat']:       
        #for use_model in ['lat', 'reg0','svd3_reg10.0_06_05']: # 'hesslat',  'combmodellat', 'combhesslat']:       
        for use_model in ['svd_reg1_06_05', 'logdet_reg1_06_05']: # 'hesslat',  'combmodellat', 'combhesslat']:       
            if use_model == 'lat':
                print("latent only\n")
                latent = torch.load(os.path.join(args.latcode_folder, 'test_latent_'+args.latfname+'_'+fname+'.py'))
                samples,_, threshcurv, meancurv, mediancurv = getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, basemodel, -1, 300000, args, 'reconstruct_lat_'+fname)
                lat_chamdist[fname] = chamfer(samples, basefname, args)
                lat_discthresh[fname] = threshcurv[0]
                lat_impthresh[fname] = threshcurv[1]
                lat_discmeancurv[fname] = meancurv[0]
                lat_impmeancurv[fname] = meancurv[1]
                lat_discmediancurv[fname] = mediancurv[0]
                lat_impmediancurv[fname] = mediancurv[1]
            else:
                model, latent  = loadEvaluateModel(None, use_model+'_'+fname, args, args.checkpoint_folder)
                if model is None:
                    return
                model.to(device)
                print("loaded evaluation {} model".format(use_model))
                samples, _, threshcurv, meancurv, mediancurv = getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 300000, args, 'reconstruct_'+use_model+'_'+fname)
                if len(samples) <= 0:
                    print(fname +" "+use_model+" IF level is 0.. rerun")
                    continue
                chamdist = chamfer(samples, basefname, args)

                if use_model == 'reg0':
                    reg0_chamdist[fname] = chamdist
                    reg0_discthresh[fname] = threshcurv[0]
                    reg0_impthresh[fname] = threshcurv[1]
                    reg0_discmeancurv[fname] = meancurv[0]
                    reg0_impmeancurv[fname] = meancurv[1]
                    reg0_discmediancurv[fname] = mediancurv[0]
                    reg0_impmediancurv[fname] = mediancurv[1]
                elif use_model == 'svd3_reg10.0_06_05':
                    svd3_reg1e1_chamdist[fname] = chamdist
                    svd3_reg1e1_discthresh[fname] = threshcurv[0]
                    svd3_reg1e1_impthresh[fname] = threshcurv[1]
                    svd3_reg1e1_discmeancurv[fname] = meancurv[0]
                    svd3_reg1e1_impmeancurv[fname] = meancurv[1]
                    svd3_reg1e1_discmediancurv[fname] = mediancurv[0]
                    svd3_reg1e1_impmediancurv[fname] = mediancurv[1]


    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_impmeancurv.json'),'w') as fwrite:
        json.dump(lat_impmeancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_impmeancurv.json'),'w') as fwrite:
        json.dump(reg0_impmeancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.0_06_05_impmeancurv.json'),'w') as fwrite:
        json.dump(svd3_reg1e1_impmeancurv, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_impmediancurv.json'),'w') as fwrite:
        json.dump(lat_impmediancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_impmediancurv.json'),'w') as fwrite:
        json.dump(reg0_impmediancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.0_06_05_impmediancurv.json'),'w') as fwrite:
        json.dump(svd3_reg1e1_impmediancurv, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_impthresh.json'),'w') as fwrite:
        json.dump(lat_impthresh, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_impthresh.json'),'w') as fwrite:
        json.dump(reg0_impthresh, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.0_06_05_impthresh.json'),'w') as fwrite:
        json.dump(svd3_reg1e1_impthresh, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_chamdist.json'),'w') as fwrite:
        json.dump(lat_chamdist, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_chamdist.json'),'w') as fwrite:
        json.dump(reg0_chamdist, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.0_06_05_chamdist.json'),'w') as fwrite:
        json.dump(svd3_reg1e1_chamdist, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_discmeancurv.json'),'w') as fwrite:
        json.dump(lat_discmeancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_discmeancurv.json'),'w') as fwrite:
        json.dump(reg0_discmeancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.0_06_05_discmeancurv.json'),'w') as fwrite:
        json.dump(svd3_reg1e1_discmeancurv, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_discmediancurv.json'),'w') as fwrite:
        json.dump(lat_discmediancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_discmediancurv.json'),'w') as fwrite:
        json.dump(reg0_discmediancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.0_06_05_discmediancurv.json'),'w') as fwrite:
        json.dump(svd3_reg1e1_discmediancurv, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_discthresh.json'),'w') as fwrite:
        json.dump(lat_discthresh, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_discthresh.json'),'w') as fwrite:
        json.dump(reg0_discthresh, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg10.0_06_05_discthresh.json'),'w') as fwrite:
        json.dump(svd3_reg1e1_discthresh, fwrite)

    avg_lat_impmedian = np.array(list(lat_impmediancurv.values())).mean()
    avg_reg0_impmedian = np.array(list(reg0_impmediancurv.values())).mean()
    avg_svd3_reg1e1_impmedian = np.array(list(svd3_reg1e1_impmediancurv.values())).mean()

    avg_lat_impmean = np.array(list(lat_impmeancurv.values())).mean()
    avg_reg0_impmean = np.array(list(reg0_impmeancurv.values())).mean()
    avg_svd3_reg1e1_impmean = np.array(list(svd3_reg1e1_impmeancurv.values())).mean()

    avg_lat_impthresh = 100*np.array(list(lat_impthresh.values())).mean()
    avg_reg0_impthresh = 100*np.array(list(reg0_impthresh.values())).mean()
    avg_svd3_reg1e1_impthresh = 100*np.array(list(svd3_reg1e1_impthresh.values())).mean()

    avg_lat_discmedian = np.array(list(lat_discmediancurv.values())).mean()
    avg_reg0_discmedian = np.array(list(reg0_discmediancurv.values())).mean()
    avg_svd3_reg1e1_discmedian = np.array(list(svd3_reg1e1_discmediancurv.values())).mean()

    avg_lat_discmean = np.array(list(lat_discmeancurv.values())).mean()
    avg_reg0_discmean = np.array(list(reg0_discmeancurv.values())).mean()
    avg_svd3_reg1e1_discmean = np.array(list(svd3_reg1e1_discmeancurv.values())).mean()

    avg_lat_discthresh = 100*np.array(list(lat_discthresh.values())).mean()
    avg_reg0_discthresh = 100*np.array(list(reg0_discthresh.values())).mean()
    avg_svd3_reg1e1_discthresh = 100*np.array(list(svd3_reg1e1_discthresh.values())).mean()

    avg_lat_chamdist = np.array(list(lat_chamdist.values())).mean()
    avg_reg0_chamdist = np.array(list(reg0_chamdist.values())).mean()
    avg_svd3_reg1e1_chamdist = np.array(list(svd3_reg1e1_chamdist.values())).mean()

    index += 1
#    print("number files = ", index)
    print("average lat imp median ", avg_lat_impmedian)
    print("average lat disc median ", avg_lat_discmedian)
    print("\n")
    print("average reg0 imp median ", avg_reg0_impmedian)
    print("average reg0 disc median ", avg_reg0_discmedian)
    print("\n")
    print("average svd3_reg1e1 imp median ", avg_svd3_reg1e1_impmedian)
    print("average svd3_reg1e1 disc median ", avg_svd3_reg1e1_discmedian)
    print("------------------------\n")
    print("average lat imp mean ", avg_lat_impmean)
    print("average lat disc mean ", avg_lat_discmean)
    print("\n")
    print("average reg0 imp mean ", avg_reg0_impmean)
    print("average reg0 disc mean ", avg_reg0_discmean)
    print("\n")
    print("average svd3_reg1e1 imp mean ", avg_svd3_reg1e1_impmean)
    print("average svd3_reg1e1 disc mean ", avg_svd3_reg1e1_discmean)
    print("------------------------\n")
    print("average lat imp thresh ", avg_lat_impthresh)
    print("average lat disc thresh ", avg_lat_discthresh)
    print("\n")
    print("average reg0 imp thresh ", avg_reg0_impthresh)
    print("average reg0 disc thresh ", avg_reg0_discthresh)
    print("\n")
    print("average svd3_reg1e1 imp thresh ", avg_svd3_reg1e1_impthresh)
    print("average svd3_reg1e1 disc thresh ", avg_svd3_reg1e1_discthresh)
    print("------------------------\n")
    print("average lat chamdist ", avg_lat_chamdist)
    print("average reg0 chamdist ", avg_reg0_chamdist)
    print("average svd3_reg1e1 chamdist ", avg_svd3_reg1e1_chamdist)
    print("------------------------\n")

def reconstructvis(args):
    grid_uniformsamples = gridData(args=args)
    basemodel, _,_, _ = loadPretrainedModel(None, args)
    basemodel.to(device)

    alltestfile = open(args.testfilepath,'r')
    lat_impthresh, reg0_impthresh, svd3_reg1e1_impthresh = {}, {}, {}
    lat_impmeancurv, reg0_impmeancurv, svd3_reg1e1_impmeancurv = {}, {}, {}
    lat_impmediancurv, reg0_impmediancurv, svd3_reg1e1_impmediancurv = {}, {}, {}
    lat_discthresh, reg0_discthresh, svd3_reg1e1_discthresh = {}, {}, {}
    lat_discmeancurv, reg0_discmeancurv, svd3_reg1e1_discmeancurv = {}, {}, {}
    lat_discmediancurv, reg0_discmediancurv, svd3_reg1e1_discmediancurv = {}, {}, {}
    lat_chamdist, reg0_chamdist, svd3_reg1e1_chamdist = {}, {}, {}
    index = 0
    for fname in alltestfile:
        fname = fname.strip()
        basefname = fname[11:]

        #for use_model in ['lat','reg0.0', 'reg10.0']: # 'hesslat',  'combmodellat', 'combhesslat']:       
        for use_model in ['lat','reg0', 'reg10.0']: # 'hesslat',  'combmodellat', 'combhesslat']:       
            if use_model == 'lat':
                print("latent only\n")
                latent = torch.load(os.path.join(args.latcode_folder, 'test_latent_'+fname+'_'+args.latfname+'_'+fname+'.py'))
                samples,_, threshcurv, meancurv, mediancurv = getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, basemodel, -1, 300000, args, 'reconstruct_lat_'+fname)
                lat_chamdist[fname] = chamfer(samples, basefname, args)
                lat_discthresh[fname] = threshcurv[0]
                lat_impthresh[fname] = threshcurv[1]
                lat_discmeancurv[fname] = meancurv[0]
                lat_impmeancurv[fname] = meancurv[1]
                lat_discmediancurv[fname] = mediancurv[0]
                lat_impmediancurv[fname] = mediancurv[1]
            else:
                model, latent  = loadEvaluateModel(None, use_model+'_'+fname, args, )
                if model is None:
                    return
                model.to(device)
                print("loaded evaluation {} model".format(use_model))
                samples, _, threshcurv, meancurv, mediancurv = getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 300000, args, 'reconstruct_'+use_model+'_'+fname)
                if len(samples) <= 0:
                    print(fname +" "+use_model+" IF level is 0.. rerun")
                    continue
                chamdist = chamfer(samples, basefname, args)

                if use_model == 'reg0':
                    reg0_chamdist[fname] = chamdist
                    reg0_discthresh[fname] = threshcurv[0]
                    reg0_impthresh[fname] = threshcurv[1]
                    reg0_discmeancurv[fname] = meancurv[0]
                    reg0_impmeancurv[fname] = meancurv[1]
                    reg0_discmediancurv[fname] = mediancurv[0]
                    reg0_impmediancurv[fname] = mediancurv[1]
                elif use_model == 'reg10.0':
                    svd3_reg1e1_chamdist[fname] = chamdist
                    svd3_reg1e1_discthresh[fname] = threshcurv[0]
                    svd3_reg1e1_impthresh[fname] = threshcurv[1]
                    svd3_reg1e1_discmeancurv[fname] = meancurv[0]
                    svd3_reg1e1_impmeancurv[fname] = meancurv[1]
                    svd3_reg1e1_discmediancurv[fname] = mediancurv[0]
                    svd3_reg1e1_impmediancurv[fname] = mediancurv[1]


        with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_impmeancurv.json'),'w') as fwrite:
            json.dump(lat_impmeancurv, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_impmeancurv.json'),'w') as fwrite:
            json.dump(reg0_impmeancurv, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1e1_impmeancurv.json'),'w') as fwrite:
            json.dump(svd3_reg1e1_impmeancurv, fwrite)

        with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_impmediancurv.json'),'w') as fwrite:
            json.dump(lat_impmediancurv, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_impmediancurv.json'),'w') as fwrite:
            json.dump(reg0_impmediancurv, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1e1_impmediancurv.json'),'w') as fwrite:
            json.dump(svd3_reg1e1_impmediancurv, fwrite)

        with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_impthresh.json'),'w') as fwrite:
            json.dump(lat_impthresh, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_impthresh.json'),'w') as fwrite:
            json.dump(reg0_impthresh, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1e1_impthresh.json'),'w') as fwrite:
            json.dump(svd3_reg1e1_impthresh, fwrite)

        with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_chamdist.json'),'w') as fwrite:
            json.dump(lat_chamdist, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_chamdist.json'),'w') as fwrite:
            json.dump(reg0_chamdist, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1e1_chamdist.json'),'w') as fwrite:
            json.dump(svd3_reg1e1_chamdist, fwrite)

        with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_discmeancurv.json'),'w') as fwrite:
            json.dump(lat_discmeancurv, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_discmeancurv.json'),'w') as fwrite:
            json.dump(reg0_discmeancurv, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1e1_discmeancurv.json'),'w') as fwrite:
            json.dump(svd3_reg1e1_discmeancurv, fwrite)

        with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_discmediancurv.json'),'w') as fwrite:
            json.dump(lat_discmediancurv, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_discmediancurv.json'),'w') as fwrite:
            json.dump(reg0_discmediancurv, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1e1_discmediancurv.json'),'w') as fwrite:
            json.dump(svd3_reg1e1_discmediancurv, fwrite)

        with open(os.path.join(args.testoutputdir, args.testfilename+'_lat_discthresh.json'),'w') as fwrite:
            json.dump(lat_discthresh, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_reg0_discthresh.json'),'w') as fwrite:
            json.dump(reg0_discthresh, fwrite)
        with open(os.path.join(args.testoutputdir, args.testfilename+'_svd3_reg1e1_discthresh.json'),'w') as fwrite:
            json.dump(svd3_reg1e1_discthresh, fwrite)

        avg_lat_impmedian = np.array(list(lat_impmediancurv.values())).mean()
        avg_reg0_impmedian = np.array(list(reg0_impmediancurv.values())).mean()
        avg_svd3_reg1e1_impmedian = np.array(list(svd3_reg1e1_impmediancurv.values())).mean()

        avg_lat_impmean = np.array(list(lat_impmeancurv.values())).mean()
        avg_reg0_impmean = np.array(list(reg0_impmeancurv.values())).mean()
        avg_svd3_reg1e1_impmean = np.array(list(svd3_reg1e1_impmeancurv.values())).mean()

        avg_lat_impthresh = 100*np.array(list(lat_impthresh.values())).mean()
        avg_reg0_impthresh = 100*np.array(list(reg0_impthresh.values())).mean()
        avg_svd3_reg1e1_impthresh = 100*np.array(list(svd3_reg1e1_impthresh.values())).mean()

        avg_lat_discmedian = np.array(list(lat_discmediancurv.values())).mean()
        avg_reg0_discmedian = np.array(list(reg0_discmediancurv.values())).mean()
        avg_svd3_reg1e1_discmedian = np.array(list(svd3_reg1e1_discmediancurv.values())).mean()

        avg_lat_discmean = np.array(list(lat_discmeancurv.values())).mean()
        avg_reg0_discmean = np.array(list(reg0_discmeancurv.values())).mean()
        avg_svd3_reg1e1_discmean = np.array(list(svd3_reg1e1_discmeancurv.values())).mean()

        avg_lat_discthresh = 100*np.array(list(lat_discthresh.values())).mean()
        avg_reg0_discthresh = 100*np.array(list(reg0_discthresh.values())).mean()
        avg_svd3_reg1e1_discthresh = 100*np.array(list(svd3_reg1e1_discthresh.values())).mean()

        avg_lat_chamdist = np.array(list(lat_chamdist.values())).mean()
        avg_reg0_chamdist = np.array(list(reg0_chamdist.values())).mean()
        avg_svd3_reg1e1_chamdist = np.array(list(svd3_reg1e1_chamdist.values())).mean()

        index += 1
        print("number files = ", index)
        print("average lat imp median ", avg_lat_impmedian)
        print("average lat disc median ", avg_lat_discmedian)
        print("\n")
        print("average reg0 imp median ", avg_reg0_impmedian)
        print("average reg0 disc median ", avg_reg0_discmedian)
        print("\n")
        print("average svd3_reg1e1 imp median ", avg_svd3_reg1e1_impmedian)
        print("average svd3_reg1e1 disc median ", avg_svd3_reg1e1_discmedian)
        print("------------------------\n")
        print("average lat imp mean ", avg_lat_impmean)
        print("average lat disc mean ", avg_lat_discmean)
        print("\n")
        print("average reg0 imp mean ", avg_reg0_impmean)
        print("average reg0 disc mean ", avg_reg0_discmean)
        print("\n")
        print("average svd3_reg1e1 imp mean ", avg_svd3_reg1e1_impmean)
        print("average svd3_reg1e1 disc mean ", avg_svd3_reg1e1_discmean)
        print("------------------------\n")
        print("average lat imp thresh ", avg_lat_impthresh)
        print("average lat disc thresh ", avg_lat_discthresh)
        print("\n")
        print("average reg0 imp thresh ", avg_reg0_impthresh)
        print("average reg0 disc thresh ", avg_reg0_discthresh)
        print("\n")
        print("average svd3_reg1e1 imp thresh ", avg_svd3_reg1e1_impthresh)
        print("average svd3_reg1e1 disc thresh ", avg_svd3_reg1e1_discthresh)
        print("------------------------\n")
        print("average lat chamdist ", avg_lat_chamdist)
        print("average reg0 chamdist ", avg_reg0_chamdist)
        print("average svd3_reg1e1 chamdist ", avg_svd3_reg1e1_chamdist)
        print("------------------------\n")

def chamfer(reconstructpoints, fname, args):
    print("fname = ",fname)
    pointnormal = getPointNormal(fname, args.onsurfdir)
    gt_points = normalize_pts_withdia(pointnormal[:, :3])
    reconstructpoints = normalize_pts_withdia(reconstructpoints)
    #convertToPLY(torch.tensor(gt_points), None, None, True, 'gt_tanh_svd3_10'+fname)
    #convertToPLY(torch.tensor(reconstructpoints), None, None, True, 'reconst_tanh_svd3_10'+fname)
#    maxdim = np.max(gt_points, axis=0)
#    mindim = np.min(gt_points, axis=0)
#    print("maxdim = ", maxdim)
#    print("mindim = ", mindim)
#    maxdim = np.max(reconstructpoints, axis=0)
#    mindim = np.min(reconstructpoints, axis=0)
#    print("re maxdim = ", maxdim)
#    print("re mindim = ", mindim)
    dist = getChamferDist(gt_points, reconstructpoints)
    print('chamfer dist = ',dist, flush=True)
    return dist

