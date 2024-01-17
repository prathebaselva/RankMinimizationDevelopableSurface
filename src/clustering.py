import os
import open3d as o3d
from dataset import gridData
from initialize import initDeepsdfTestDataSet, initlatentOptimizer, initlatcodeandmodelOptimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loadmodel import initModel, loadPretrainedModel, loadCheckpointModel
from loss import datafidelity_testloss, implicit_loss
from utils import mkdir_p,save_reg0checkpoint, save_checkpoint, save_curr_checkpoint, plotgauss, plotloss
from trainhelper import getSurfaceSamplePoints, getDiscAndImpCurvatureOfSurface
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch
import numpy as np
from gradient import getGradientAndHessian
import shutil

deviceids = []
if cuda.is_available():
    for i in range(cuda.device_count()):
        deviceids.append(i)
    cuda.set_device(0)
device = torch.device("cuda" if cuda.is_available() else "cpu")


def loadCheckpointModelHelper(model, args, use_model):
    print("\nloading training checkpoint model", use_model)
    path_to_resume_file = os.path.join(args.checkpoint_folder, 'model_best_'+use_model+'.pth.tar')
    if os.path.exists(path_to_resume_file):
        print("=> Loading training checkpoint '{}'".format(path_to_resume_file))
        checkpoint = torch.load(path_to_resume_file)
        model.load_state_dict(checkpoint['state_dict'])
        if 'train_latent' in checkpoint:
            latent = checkpoint['train_latent']
        elif 'test_latent' in checkpoint:
            latent = checkpoint['test_latent']
        best_loss = checkpoint['best_loss'] 
        args.start_epoch = checkpoint['epoch']
        args.lr = checkpoint['lr']
        return model, latent, best_loss, args
    else:
        print( "No training checkpoint model exists")
        return None, None, None, args


def retrainlatCodeandModelOptimization(args,filename, usemodel,hess_delta):
    model = initModel(args)
    cudnn.benchmark = True
     
    test_dataset = initDeepsdfTestDataSet(args, filename)
    grid_uniformsamples = gridData(args=args)
    indexfname_dict = test_dataset.getindexfname()

    test_latent = {}
    for index, fname in indexfname_dict.items():
        args.use_model = usemodel+'_'+fname
        model, latent, best_loss  = loadCheckpointModel(model, args)
        if model is None:
            return
        model.to(device)
        model.eval()
        print("loaded pretrained model")
        test_latent[fname] = torch.ones(args.lat).normal_(mean=0, std=0.01).to(device)
        test_latent[fname].data = latent.data.clone()
        test_latent[fname].to(device)
        test_latent[fname].requires_grad = True
        print("test_latent=", test_latent[fname][0:10], flush=True)
        #args.code_delta = 1e-07 
        #args.latlr = 1e-05
        latcodeandmodeloptimizer = initlatcodeandmodelOptimizer(model, test_latent[fname], args)
        scheduler = ReduceLROnPlateau(latcodeandmodeloptimizer, factor=0.9,patience=15,mode='min',threshold=1e-4, eps=0, min_lr=0)

        getlatCodeAndModel(index, test_dataset, grid_uniformsamples, test_latent[fname],  model, latcodeandmodeloptimizer , scheduler, args, 'remodel_best_reg'+str(hess_delta)+'_'+fname, 'bestremodelreg'+str(hess_delta)+'_'+fname, best_loss)


def latCodeOptimization(args, filename):
    model = initModel(args)
    cudnn.benchmark = True

    model, _,_, _ = loadPretrainedModel(model, args)
    if model is None:
        return
    model.to(device)
    model.eval()
    print("loaded pretrained model")
    test_dataset = initDeepsdfTestDataSet(args, filename)
    grid_uniformsamples = gridData(args=args)

    indexfname_dict = test_dataset.getindexfname()
    test_latent = {} 
    if not os.path.isdir(args.latcode_folder):
        print("Creating new latcode folder path" + args.latcode_folder)
        mkdir_p(args.latcode_folder)

    for index,fname in indexfname_dict.items():
        if(os.path.exists(os.path.join(args.latcode_folder, 'test_latent_'+args.latfname+'_'+fname+'.py'))):
            latent = torch.load(os.path.join(args.latcode_folder, 'test_latent_'+args.latfname+'_'+fname+'.py'))
            getDiscAndImpCurvatureOfSurface(grid_uniformsamples, latent, model, -1, 300000, args, 'bestlatcode_'+fname)
            test_latent[fname] = latent
        else:
            test_latent[fname] = torch.ones(args.lat).normal_(mean=0, std=0.01).to(device)
            test_latent[fname].requires_grad = True
            print("test_latent=", test_latent[fname][0:10], flush=True)

            latoptimizer = initlatentOptimizer(test_latent[fname], args)
            latscheduler = ReduceLROnPlateau(latoptimizer, factor=0.9,patience=15,mode='min',threshold=1e-4, eps=0, min_lr=0)

            loss, test_latent[fname] = getlatCode(index, test_dataset, test_latent[fname], model, latoptimizer, latscheduler, args, 'model_best_lat_'+fname, 'bestmodellat_'+fname)
            torch.save(test_latent[fname], os.path.join(args.latcode_folder, 'test_latent_'+args.latfname+'_'+fname+'.py'))
            getDiscAndImpCurvatureOfSurface(grid_uniformsamples, test_latent[fname], model, -1, 300000, args, 'bestlatcode_'+fname)

    #return samples
def latCodeandModelOptimization(args,filename, hess_delta):
    model = initModel(args)
    cudnn.benchmark = True

    model, _,_, _ = loadPretrainedModel(model, args)
    if model is None:
        return
    model.to(device)
    model.eval()
    print("loaded pretrained model")
    test_dataset = initDeepsdfTestDataSet(args, filename)
    grid_uniformsamples = gridData(args=args)
    indexfname_dict = test_dataset.getindexfname()

    test_latent = {}
    for index, fname in indexfname_dict.items():
        baselatpath = os.path.join(args.latcode_folder,'test_latent_'+args.latfname+'_'+fname+'.py')
        if not os.path.exists(baselatpath):
            latent = latCodeOptimization(args)
        latent = torch.load(baselatpath)
        test_latent[fname] = torch.ones(args.lat).normal_(mean=0, std=0.01).to(device)
        test_latent[fname].data = latent.data.clone()
        test_latent[fname].to(device)
        test_latent[fname].requires_grad = True
        print("test_latent=", test_latent[fname][0:10], flush=True)
        #args.code_delta = 1e-07 
        #args.latlr = 1e-05
        latcodeandmodeloptimizer = initlatcodeandmodelOptimizer(model, test_latent[fname], args)
        scheduler = ReduceLROnPlateau(latcodeandmodeloptimizer, factor=0.9,patience=15,mode='min',threshold=1e-4, eps=0, min_lr=0)

        getlatCodeAndModel(index, test_dataset, grid_uniformsamples, test_latent[fname],  model, latcodeandmodeloptimizer , scheduler, args, 'model_best_reg'+str(hess_delta)+'_'+fname, 'bestmodelreg'+str(hess_delta)+'_'+fname)

def getlatCode(index, test_dataset, latent, model, optimizer, scheduler, args, modelname='model_best_lat', reconstrcutname='bestmodellat'):
    #for i in range(num_batch):
    best_loss = 2e10
    model.eval()
    #for epoch in range(1200):
    for epoch in range(1200):
        loss_sum = 0.0
        loss_count = 0.0

        for chunk in range(550):
            fileindex, data = test_dataset[index]  # a dict
            optimizer.zero_grad()
            sampled_points = data[:,0:3].to(device) # data['xyz'].to(device)

            this_bs =  sampled_points.shape[0]
            lat_vec = latent.expand(this_bs,-1).to(device)
              
            sampled_points = torch.cat([lat_vec, sampled_points],dim=1).to(device)
            predicted_sdf = model(sampled_points)
            gt_sdf_tensor = torch.clamp(data[:,3:].to(device), -args.clamping_distance, args.clamping_distance)

            loss = datafidelity_testloss(predicted_sdf, gt_sdf_tensor, latent, epoch,  args)
            printinterval = 0
            #if chunk == 1:
            #   print("data fidelity loss = ",loss, flush=True)

            loss.backward()
            loss_sum += loss.item() * this_bs
            loss_count += this_bs
            optimizer.step()
            
        epochloss = loss_sum / loss_count
        if epochloss < best_loss: 
            best_loss = epochloss
            best_latent = latent.clone()
            print("Best epoch::loss = {}::{} ".format(epoch, epochloss), flush=True)
            #print(latent[0:10], flush=True)
            for param_group in optimizer.param_groups:
                print("LR step :: ",param_group['lr'])
                curr_lr = param_group['lr']
            save_reg0checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict(), "test_latent":latent}, True, checkpoint_folder=args.checkpoint_folder, modelname=modelname)
        scheduler.step(epochloss) 
    save_curr_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict(), "test_latent":latent}, checkpoint_folder=args.checkpoint_folder,modelname=modelname)
    return best_loss, best_latent


def getlatCodeAndModel(index, test_dataset, grid_uniformsamplepoints, latent, model, optimizer, scheduler, args, modelname='model_best_modellat', reconstructname='bestmodellatcode', best_loss=2e10):
    #for i in range(num_batch)::w!

    model.eval()
    mcube_points = getSurfaceSamplePoints(grid_uniformsamplepoints, latent, model, -1, 300000, args, reconstructname)
    all_gausscurv = [[],[],[],[]]
    best_loss = 10000
    
    #for epoch in range(600):
    for epoch in range(1000):
        loss_sum = 0.0
        loss_count = 0.0
        regloss_sum = 0.0
        sdfloss_sum = 0.0

        for chunk in range(550):
            fileindex, data = test_dataset[index]  # a dict
            optimizer.zero_grad()
            sampled_points = data[:,0:3].to(device) # data['xyz'].to(device)

            this_bs =  sampled_points.shape[0]
            lat_vec = latent.expand(this_bs,-1).to(device)
              
            sampled_points = torch.cat([lat_vec, sampled_points],dim=1).to(device)
            predicted_sdf = model(sampled_points)
            gt_sdf_tensor = torch.clamp(data[:,3:].to(device), -args.clamping_distance, args.clamping_distance)

            sdfloss = datafidelity_testloss(predicted_sdf, gt_sdf_tensor, latent, epoch,  args)
            printinterval = 0
           # if chunk == 1:
           #     print("data fidelity loss = ",sdfloss, flush=True)
           #     printinterval = 1
            if torch.isnan(sdfloss):
                scheduler.step(2e20) 
                continue
  

            mcube_r = np.random.randint( len(mcube_points), size=(this_bs,)) 
            mcube_sampled_points =  torch.tensor(mcube_points[mcube_r]).to(device)
            this_bs =  mcube_sampled_points.shape[0]
            mcube_sampled_points.requires_grad = True
            lat_mcube_sampled_points = torch.cat([lat_vec, mcube_sampled_points],dim=1).to(device)
            mcube_predicted_sdf = model(lat_mcube_sampled_points.float())
            hess_indices = np.arange(this_bs) # torch.where((torch.abs(mcube_predicted_sdf) <= args.threshold))[0]
            surfaceP_gradient,surfaceP_hessian_matrix  = getGradientAndHessian(mcube_predicted_sdf, mcube_sampled_points)
            regloss = implicit_loss(epoch, printinterval, surfaceP_gradient, surfaceP_hessian_matrix, None , args, hess_indices, mcube_predicted_sdf)
            if torch.isnan(regloss):
                scheduler.step(2e20) 
                continue

            #if printinterval:
            #    print("regularizer loss = ",regloss)
            loss = sdfloss + regloss

            loss.backward()
            loss_sum += loss.item() * this_bs
            loss_count += this_bs
            optimizer.step()
            
        epochloss = loss_sum / loss_count
        if epoch % 5 == 0:
            newmcube_points = getSurfaceSamplePoints(grid_uniformsamplepoints, latent, model, epoch, 300000, args, reconstructname)

            if newmcube_points is not None:
                mcube_points = newmcube_points.copy()
        if epochloss < best_loss:
            best_epoch = epoch
            best_loss = epochloss
            print("Best Epoch::loss {}::{} ".format(epoch, epochloss),flush=True)
            #print("latent = ", latent[0:10], flush=True)
            shutil.copy(os.path.join(args.outfolder, args.save_file_name+'_'+reconstructname+'.obj'), os.path.join(args.outfolder, args.save_file_name+'_'+reconstructname+'loss.obj'))
            for param_group in optimizer.param_groups: 
                print("LR step :: ",param_group['lr']) 
                curr_lr = param_group['lr']
            best_latent = latent.clone()
            save_reg0checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict(), "test_latent":latent}, True, checkpoint_folder=args.checkpoint_folder, modelname=modelname)
        scheduler.step(epochloss) 
    save_curr_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict(), "test_latent":latent}, checkpoint_folder=args.checkpoint_folder,modelname=modelname)
          
    return 

    best_loss = 2e10

    model.eval()
    mcube_points = getSurfaceSamplePoints(grid_uniformsamplepoints, latent, model, -1, 300000, args, 'bestmodellatcode')
    best_data = np.array([1000,1000,1000,1000,1000])
    best_latent = np.array([1000,1000,1000,1000,1000])
    all_gausscurv = [[],[],[],[]]
    for epoch in range(1200):
        loss_sum = 0.0
        loss_count = 0.0
        regloss_sum = 0.0
        sdfloss_sum = 0.0

        for chunk in range(550):
            fileindex, data = test_dataset[index]  # a dict
            optimizer.zero_grad()
            sampled_points = data[:,0:3].to(device) # data['xyz'].to(device)

            this_bs =  sampled_points.shape[0]
            lat_vec = latent.expand(this_bs,-1).to(device)
              
            sampled_points = torch.cat([lat_vec, sampled_points],dim=1).to(device)
            predicted_sdf = model(sampled_points)
            gt_sdf_tensor = torch.clamp(data[:,3:].to(device), -args.clamping_distance, args.clamping_distance)

            sdfloss = datafidelity_testloss(predicted_sdf, gt_sdf_tensor, latent, epoch,  args)
            printinterval = 0
            if chunk == 1:
                print("data fidelity loss = ",sdfloss, flush=True)
                printinterval = 1
            mcube_r = np.random.randint( len(mcube_points), size=(this_bs,)) 
            mcube_sampled_points =  torch.tensor(mcube_points[mcube_r]).to(device)
            this_bs =  mcube_sampled_points.shape[0]
            mcube_sampled_points.requires_grad = True
            lat_mcube_sampled_points = torch.cat([lat_vec, mcube_sampled_points],dim=1).to(device)
            mcube_predicted_sdf = model(lat_mcube_sampled_points.float())
            hess_indices = np.arange(this_bs) # torch.where((torch.abs(mcube_predicted_sdf) <= args.threshold))[0]
            surfaceP_gradient,surfaceP_hessian_matrix  = getGradientAndHessian(mcube_predicted_sdf, mcube_sampled_points)
            regloss = implicit_loss(epoch, printinterval, surfaceP_gradient, surfaceP_hessian_matrix, None , args, hess_indices, mcube_predicted_sdf)

            if printinterval:
                print("regularizer loss = ",regloss)
            loss = sdfloss + regloss

            loss.backward()
            loss_sum += loss.item() * this_bs
            loss_count += this_bs
            optimizer.step()
            
        epochloss = loss_sum / loss_count
        newmcube_points, best_data, best_bool, all_gausscurv= getnewmcubepoints(grid_uniformsamplepoints, latent, model, epoch,epochloss,best_data, all_gausscurv, args,  reconstructname) 

        if newmcube_points is not None:
            mcube_points = newmcube_points.copy()
        if best_bool.any():
            best_epoch = epoch
            print("latent = ", latent[0:10], flush=True)
            for param_group in optimizer.param_groups: 
                print("LR step :: ",param_group['lr']) 
                curr_lr = param_group['lr']
            save_checkpoint({"epoch": epoch ,"test_latent":latent, "lr": curr_lr, "state_dict": model.state_dict(), "best_discmean": best_data[0], "best_discmedian": best_data[1], "best_impmean": best_data[2], "best_impmedian": best_data[3], "best_loss": best_data[4], "optimizer": optimizer.state_dict()}, best_bool[0], best_bool[1], best_bool[2], best_bool[3], best_bool[4], checkpoint_folder=args.checkpoint_folder, modelname=modelname)

        #scheduler.step(mcube_impmedian)
        scheduler.step(epochloss)
        save_curr_checkpoint({"epoch": epoch ,"test_latent":latent, "lr": curr_lr, "state_dict": model.state_dict(), "best_discmean": best_data[0], "best_discmedian": best_data[1], "best_impmean": best_data[2], "best_impmedian": best_data[3], "best_loss": best_data[4], "optimizer": optimizer.state_dict()}, checkpoint_folder=args.checkpoint_folder, modelname=modelname) 
          
    return 

def getnewmcubepoints(validate_dataset, latent, model, epoch, epochloss, best_data, all_gausscurv, args, fname='bestmodellat'):

        is_best_discmean = False 
        is_best_discmedian = False
        is_best_impmean = False
        is_best_impmedian = False
        is_best_loss = False
        mcube_points = None
        newmcube_gaussmean = [1000, 1000]
        newmcube_gaussmedian = [1000, 1000]
        discmean, discmedian, impmean, impmedian = 0, 0, 0, 0
        newmcube_points, newmcube_surf_area, newmcube_gaussmean, newmcube_gaussmedian, newmcube_sdf = getDiscAndImpCurvatureOfSurface(validate_dataset, latent, model, epoch, 500000, args, "resamp2")
        all_gausscurv[0].append(newmcube_gaussmean[0])
        all_gausscurv[1].append(newmcube_gaussmedian[0])
        all_gausscurv[2].append(newmcube_gaussmean[1])
        all_gausscurv[3].append(newmcube_gaussmedian[1])
        plotgauss(args.outfolder, epoch, args.save_file_name+'_'+fname, all_gausscurv[0])
        plotgauss(args.outfolder, epoch, args.save_file_name+'_'+fname, all_gausscurv[1])
        plotgauss(args.outfolder, epoch, args.save_file_name+'_'+fname, all_gausscurv[2])
        plotgauss(args.outfolder, epoch, args.save_file_name+'_'+fname, all_gausscurv[3])
        if (not newmcube_points is None) and len(newmcube_points)> 0:
            print("updating points ", len(newmcube_points))
            mcube_points = newmcube_points.copy()
            discmean = newmcube_gaussmean[0]
            impmean = newmcube_gaussmean[1]
            discmedian = newmcube_gaussmedian[0]
            impmedian = newmcube_gaussmedian[1]
            #chamfer_loss = getChamferDist(base_mcube_points[i], newmcube_points)
            #print("chamfer loss = ",chamfer_loss)
        else:
            discmean = 1000
            discmedian = 1000
            impmean = 1000
            impmedian = 1000
                
        is_best_discmean = discmean < best_data[0]
        is_best_discmedian = discmedian < best_data[1]
        is_best_impmean = impmean < best_data[2]
        is_best_impmedian = impmedian < best_data[3]
        is_best_loss = epochloss < best_data[4]

        if is_best_discmean:
            print("Epoch {} has best disc mean".format(epoch))
            best_data[0] = discmean
            shutil.copy(os.path.join(args.outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(args.outfolder, args.save_file_name+'_'+fname+'discmean.obj'))
        if is_best_discmedian:
            print("Epoch  {} has best disc median".format(epoch))
            best_data[1] = discmedian
            shutil.copy(os.path.join(args.outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(args.outfolder, args.save_file_name+'_'+fname+'discmedian.obj'))
        if is_best_impmean:
            print("Epoch  {} has best imp mean".format(epoch))
            best_data[2] = impmean
            shutil.copy(os.path.join(args.outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(args.outfolder, args.save_file_name+'_'+fname+'impmean.obj'))
        if is_best_impmedian:
            print("Epoch  {} has best imp median".format(epoch))
            best_data[3] = impmedian
            shutil.copy(os.path.join(args.outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(args.outfolder, args.save_file_name+'_'+fname+'impmedian.obj'))
        if is_best_loss:
            print("Epoch  {} has best loss".format(epoch))
            best_data[4] = epochloss
            shutil.copy(os.path.join(args.outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(args.outfolder, args.save_file_name+'_'+fname+'loss.obj'))

        return mcube_points, best_data, np.array([is_best_discmean, is_best_discmedian, is_best_impmean, is_best_impmedian, is_best_loss]), all_gausscurv

def clustering(pointcloud):
    o3d_pcl = o3d.geometry.PointCloud()
    o3d_pcl.points = o3d.utility.Vector3dVector(xyz)
    clusters = o3d_pcl.cluster_dbscan(eps=0.01, min_samples=10)
    print(clusters)


