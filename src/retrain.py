#import open3d
import torch.backends.cudnn as cudnn
from dataset import * # normalize_pts, normalize_normals, SdfDataset, mkdir_p, isdir
from loss import *
from curvature import *
from gradient import *
from trainhelper import *
from loadmodel import *
from utils import *
from dataset import *
from runmodel import *
from initialize import *
from getHessianMcube import *
from wnnm import *
from torch.autograd import Variable
import trimesh
import json

outfolder = '/mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/output/'
#outfolder = 'output/'

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)
   

def train(dataset,  model, optimizer, epoch, args, mcube_points=None):
    model.train()  # switch to train mode
    loss, indexcount= runmodel(dataset,  model, optimizer, epoch, args, mcube_points)
    return loss, indexcount

# validation function
def val(dataset,  model, optimizer, epoch, args, mcube_points=None):
    model.eval()  # switch to test mode
    loss, indexcount = runmodel(dataset,  model, optimizer, epoch, args, mcube_points, False)
    return loss, indexcount


def updateSamples(epoch, model, test_dataset, base_mcube_points, base_surf_area, is_best, args, count, prefname='best'):
    print("Epoch {} has best loss".format(epoch), flush=True)
    mcube_points = None
    updatevalloss = True
    comparea = base_surf_area *0.2
    if args.reg:
        if args.resamp:
            print("Epoch {} , resampling ".format(epoch))
        #newmcube_points, newsurf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, len(train_dataset)*4*args.train_batch, epoch, args)
        newmcube_points, newmcube_gauss, newsurf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, epoch, count, args, prefname)
        print("new surf area = ", newsurf_area)
        chamfer_loss = 1
        if len(newmcube_points) <= 0:
            print("surface formed not in iso level")
        else:  
            chamfer_loss = getChamferDist(base_mcube_points, newmcube_points)
            print("chamfer loss = ",chamfer_loss)

        pertarea = abs(newsurf_area - base_surf_area)/comparea 
        if pertarea > 1:
            print("surface area going up or vanishing")

        if len(newmcube_points) > 0 :# and chamfer_loss <= 5e-04 and pertarea <= 1:
            if args.resamp == 1 and is_best:
                updatemcube_points = True
                updatevalloss = True
                print("resampling...")
                mcube_points = newmcube_points
            
        else:
            updatevalloss = False
            print("Not updating the sampling points")

    return mcube_points, newmcube_gauss ,updatevalloss

def trainModel(args):
    best_loss = 2e20
    best_epoch = -1

    # create checkpoint folder
    if not isdir(args.checkpoint_folder):
        print("Creating new checkpoint folder " + args.checkpoint_folder)
        mkdir_p(args.checkpoint_folder)

    model = initModel(args)
    cudnn.benchmark = True

    if args.evaluate:
        model = loadEvaluateModel(model, args)
        if model is None:
            return
        model.to(device)
        print("loaded evaluation model")
        test_dataset = SdfDataset(phase='test', args=args)
        test(test_dataset, model, args,args.use_model)
        return

    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)       
 
    model.to(device)
    if args.reg: 
        train_dataset = initDataSetreg1(args)
    else:
        train_dataset, val_dataset = initDataSet(args)
    if args.reg:
        base_mcube_points, base_mcube_gauss, base_surf_area, base_mcube_sdf = getSurfacePoints(test_dataset, model, -1,  len(train_dataset)*args.train_batch, args)
        print("number of mcube points" , len(base_mcube_points))
        print("base surf area = ", base_surf_area)
        print("base gauss mesh curvature = ", base_mcube_gauss)

        mcube_points = base_mcube_points # getSurfacePoints(model, len(train_dataset)*4*args.train_batch, 0, args)
        mcube_sdf = base_mcube_sdf
        best_gauss = base_mcube_gauss
        #best_loss, val_indexcount = val(val_dataset, model, optimizer, -1, args, mcube_points)
        comparea = base_surf_area *0.2
        best_loss = 2e20
        n_mcube_points = len(mcube_points)
        print("=> Number of points in mcube points : %d" % n_mcube_points)

    maxsdf_count = 0
    # We do not want 512 grid size for all epochs since it will take a lot of memory and time
    test_dataset = SdfDataset(phase='test', args=args)
    optimizer = initOptimizer(model, args)
    scheduler = initScheduler(optimizer, args)
    #print("=> Will use the (" + device.type + ") device.")

    # create dataset

    allgaussHist_train = []
    allgaussHist_val = []
    all_loss_train = []
    all_loss_val = []
    all_gaussavg_train = []
    all_gaussavg_val = []
    all_gaussavg = []
    diff_epoch = 0
    curr_lr = args.lr
    batch_size = args.train_batch
    loss_epoch = 0
    indexcount_all = 0
    allavg = 0
    surfavg = 0
    resample = True
    resamplecount = 0
    numhighloss = 0
    prev_loss = 2e20

    mcube_points = []
    best_points = []

    gridsize = args.grid_N
    #args.grid_N = 1024
    #args.grid_N = 512
    test_dataset = SdfDataset(phase='test', args=args)
    print("numpoints = ", len(test_dataset))
    args.grid_N = gridsize

    best_gauss = math.inf
    
    if args.reg:
        base_mcube_points, base_mcube_gauss, base_surf_area, base_mcube_sdf = getSurfacePoints(test_dataset, model, -1,  len(train_dataset)*args.train_batch, args)
        print("number of mcube points" , len(base_mcube_points))
        print("base surf area = ", base_surf_area)
        print("base gauss mesh curvature = ", base_mcube_gauss)

        mcube_points = base_mcube_points # getSurfacePoints(model, len(train_dataset)*4*args.train_batch, 0, args)
        mcube_sdf = base_mcube_sdf
        best_gauss = base_mcube_gauss
        #best_loss, val_indexcount = val(val_dataset, model, optimizer, -1, args, mcube_points)
        comparea = base_surf_area *0.2
        best_loss = 2e20
        n_mcube_points = len(mcube_points)
        print("=> Number of points in mcube points : %d" % n_mcube_points)

    maxsdf_count = 0
    # We do not want 512 grid size for all epochs since it will take a lot of memory and time
    test_dataset = SdfDataset(phase='test', args=args)
    if args.reg == 0:
        for epoch in range(args.start_epoch, args.epochs):
            train_loss, train_maxsdf = train(train_dataset, model, optimizer, epoch, args)
            val_loss, val_maxsdf = val(val_dataset, model, optimizer, epoch, args)
            updatevalloss = True
            is_best_gauss = False

            #if epoch >= 3:
            all_loss_train.append(train_loss)
            all_loss_val.append(val_loss)
            np_all_loss_train = np.ma.masked_where(np.array(all_loss_train) >= 2e9, np.array(all_loss_train))
            np_all_loss_val = np.ma.masked_where(np.array(all_loss_val) >= 2e9, np.array(all_loss_val))

            #if epoch >= 3:
            #if epoch % 3 == 0:
            #    all_gaussavg_train.append(train_gaussavg)
            #    all_gaussavg_val.append(val_gaussavg)
            plotloss(outfolder, epoch, args.save_file_name, np_all_loss_train, np_all_loss_val)
            _,_, mcube_gauss = getGaussAvgOfSurfacePoints(test_dataset, model, epoch, len(train_dataset)*args.train_batch, args, prefname="best")
           
            all_gaussavg.append(mcube_gauss)
            plotgauss(outfolder, epoch, args.save_file_name, all_gaussavg)

            is_best = abs(val_loss) < best_loss
            loss_epoch += 1
            test_dataset = SdfDataset(phase='test', args=args)
            if is_best:
                _,mcube_gauss,_,_ = getSurfacePoints(test_dataset, model, epoch, 200000, args)
                is_best_gauss = mcube_gauss <= best_gauss


            if updatevalloss and is_best:
                loss_epoch = 0
                best_loss = val_loss
                best_epoch = epoch
                best_points = mcube_points
                if is_best_gauss:
                    best_gauss = mcube_gauss
                numhighloss = 0
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
                    curr_lr = param_group['lr']
                save_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, is_best_gauss, is_best, checkpoint_folder=args.checkpoint_folder)

            if epoch % 5 == 0:
                save_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, is_best_gauss, is_best, checkpoint_folder=args.checkpoint_folder)

            if args.scheduler == 'reducelr': 
                scheduler.step(val_loss)
            elif args.scheduler =='cosine':
                scheduler.step()

            save_curr_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, checkpoint_folder=args.checkpoint_folder)
            if epoch % 10 == 0:
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
            print(f"Epoch{epoch:d}. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}. Best Epoch: {best_epoch:d}. Best val loss: {best_loss:.8f}.",flush=True)

    if args.reg == 1:
        mcube_sdf = base_mcube_sdf.copy()
        mcube_points = base_mcube_points.copy()
        for epoch in range(args.start_epoch, args.epochs):
            is_best_gauss = False
            is_best = False
            newmcube_gauss = 1000
            #if args.altepoch2:
            #train_indexcount = 1
            #val_indexcount = 1
            #    if epoch % 2 == 0:
            #        args.mcube = 1
            #        args.data_delta = 0
            #    else:
            #        args.mcube = 0
            #        args.data_delta = 1e4
             
            if args.mcube:
                train_loss, train_maxsdf= train(train_dataset, model, optimizer, epoch, args, mcube_points)
            elif args.onsurf:
                train_loss, train_maxsdf= train(train_dataset, model, optimizer, epoch, args)
            updatevalloss = True

            #if epoch >= 3:
            all_loss_train.append(train_loss)
            np_all_loss_train = np.ma.masked_where(np.array(all_loss_train) >= 2e9, np.array(all_loss_train))
            plotloss(outfolder, epoch, args.save_file_name, np_all_loss_train)
            is_best = abs(train_loss) < best_loss

            #if args.altepoch2 and epoch % 2 == 0:
            #    is_best = abs(train_loss) < best_loss
            #elif args.altepoch2 and epoch % 2 == 1:
            #    is_best = False
            #elif not args.altepoch2:
            loss_epoch += 1

            if args.resamp==2 and args.mcube:
                newmcube_points, newmcube_gauss, newmcube_surf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, epoch, len(train_dataset)*args.train_batch, args, "resamp2")
                if (not newmcube_points is None) and len(newmcube_points)> 0:
                    print("updating points ", len(newmcube_points))
                    mcube_points = newmcube_points.copy()
                    is_best_gauss = newmcube_gauss <= best_gauss 
                else:
                    newmcube_gauss = 1000
                    is_best_gauss = False
                    is_best = False 
                all_gaussavg.append(newmcube_gauss)
                plotgauss(outfolder, epoch, args.save_file_name, all_gaussavg)

            if args.resamp==3 and args.mcube and is_best:
                newmcube_points, newmcube_gauss, newmcube_surf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, epoch, len(train_dataset)*args.train_batch, args, "resamp3")
                if (not newmcube_points is None) and len(newmcube_points)> 0:
                    mcube_points = newmcube_points.copy()
                    is_best_gauss = newmcube_gauss <= best_gauss 
                else:
                    is_best_gauss = False
                    is_best = False
                all_gaussavg.append(newmcube_gauss)
                plotgauss(outfolder, epoch, args.save_file_name, all_gaussavg)

            if args.resamp == 0 and args.mcube:
                newmcube_points, newmcube_gauss, newmcube_surf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, epoch, len(train_dataset)*args.train_batch, args, "resamp0")

            if is_best_gauss:
                #updateSamples(epoch, model, test_dataset, args)
                newmcube_points, newmcube_gauss, updatevalloss = updateSamples(epoch, model, test_dataset, base_mcube_points, base_surf_area, is_best, args, len(train_dataset)*args.train_batch, 'bestgauss')
                best_gauss = newmcube_gauss
            if is_best:
                #updateSamples(epoch, model, test_dataset, args)
                newmcube_points, newmcube_gauss, updatevalloss = updateSamples(epoch, model, test_dataset, base_mcube_points, base_surf_area, is_best, args, len(train_dataset)*args.train_batch, 'bestloss')
                best_loss = train_loss
                #if not newmcube_points is None:
                #    mcube_points = newmcube_points

            if updatevalloss and (is_best_gauss or is_best):
                loss_epoch = 0
                if is_best:
                    best_loss = train_loss
                best_epoch = epoch
                if is_best_gauss:
                    best_gauss = newmcube_gauss
                numhighloss = 0
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
                    curr_lr = param_group['lr']
                save_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_gauss":best_gauss, "best_loss": best_loss, "optimizer": optimizer.state_dict()}, is_best_gauss, is_best, checkpoint_folder=args.checkpoint_folder)

            if epoch % 5 == 0:
                save_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_gauss":best_gauss,  "best_loss": best_loss, "optimizer": optimizer.state_dict()}, is_best_gauss, is_best, checkpoint_folder=args.checkpoint_folder)

            if args.scheduler == 'reducelr': 
                scheduler.step(train_loss)
            elif args.scheduler =='cosine':
                scheduler.step()

            save_curr_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_gauss":best_gauss,  "best_loss": best_loss, "optimizer": optimizer.state_dict()}, checkpoint_folder=args.checkpoint_folder)
            if epoch % 10 == 0:
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
            print(f"Epoch{epoch:d}. train_loss: {train_loss:.8f}. Best Epoch: {best_epoch:d}. Best gauss loss: {best_gauss:.8f}, Best loss: {best_loss:.8f}.",flush=True)

