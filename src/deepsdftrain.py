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
   

def train(dataset, lat_vecs, model, optimizer, epoch, args, mcube_points=None, mcube_sdf=None):
    model.train()  # switch to train mode
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=True, drop_last=True)
    for train_data, indices in train_loader:
        data = train_data.reshape(-1, 4)
        num_samples = data.shape[0]
        points = data[:,0:3]
        gt_sdf = data[:,3:]
        loss_sum, loss_count = deepsdfrunmodel(points, gt_sdf, lat_vecs, indices, model, optimizer, epoch, args, mcube_points, mcube_sdf)
        all_loss += loss_sum
        total_count += loss_count
    loss = all_loss / loss_count

    return loss

# validation function
def val(dataset, lat_vecs, model, optimizer, epoch, args, mcube_points=None, mcube_sdf=None):
    model.eval()  # switch to test mode
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=False, drop_last=True)
    for val_data, indices in val_loader:
        data = val_data.reshape(-1, 4)
        num_samples = data.shape[0]
        points = data[:,0:3]
        gt_sdf = data[:,3:]
        loss_sum, loss_count = deepsdfrunmodel(points, gt_sdf, lat_vecs, indices, model, optimizer, epoch, args, mcube_points, mcube_sdf)
        all_loss += loss_sum
        total_count += loss_count
    loss = all_loss / loss_count
    return loss


def updateSamples(epoch, model, test_dataset, base_mcube_points, base_surf_area, is_best, args):
    print("Epoch {} has best loss".format(epoch), flush=True)
    mcube_points = None
    updatevalloss = True
    comparea = base_surf_area *0.2
    if args.reg:
        if args.resamp:
            print("Epoch {} , resampling ".format(epoch))
        #newmcube_points, newsurf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, len(train_dataset)*4*args.train_batch, epoch, args)
        newmcube_points, newsurf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, 100000, args)
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

    return mcube_points, updatevalloss

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

    if args.use_pretrained_model: 
        model = loadPretrainedModel(model, args)
        if model is None:
            return
        model.to(device)

    if args.use_checkpoint_model:
        check_model, check_best_loss, start_epoch, lr = loadCheckpointModel(model, args)
       
        if check_model:
            best_loss = check_best_loss
            args.start_epoch = start_epoch
            args.lr = lr
            model = check_model
            model.to(device)
        elif check_model is None and (args.reg and not args.use_pretrained_model):
            print("No checkpoint model exists")
            return 
        elif check_model is None and args.reg == 0:
            print("no checkpoint model exists. Start to training from scratch")
            model = initModel(args)
            model.to(device)

    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)       
 
    model.to(device)
    train_dataset = initDeepsdfDataSet(args)
    lat_vecs = torch.nn.Embedding(len(train_dataset), 256, max_norm=1.0)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        (1.0) / math.sqrt(latent_size),
    )
    optimizer = initOptimizer(model, args)
    scheduler = initScheduler(optimizer, args)
    n_points_train = int(args.train_split_ratio * len(train_dataset))
    full_indices = np.arange(len(train_dataset))
    np.random.shuffle(full_indices)
    train_indices = full_indices[:n_points_train]
    val_indices = full_indices[n_points_train:]

    allgaussHist_train = []
    allgaussHist_val = []
    all_loss_train = []
    all_loss_val = []
    all_gaussavg_train = []
    all_gaussavg_val = []
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
    args.grid_N = gridsize
    

    maxsdf_count = 0
    # We do not want 512 grid size for all epochs since it will take a lot of memory and time
    test_dataset = SdfDataset(phase='test', args=args)
    if args.reg == 0:
        for epoch in range(args.start_epoch, args.epochs):
            train_loss, train_maxsdf, train_gaussavg = train(train_dataset[train_indices], lat_vecs[train_indices], model, optimizer, epoch, args)
            val_loss, val_maxsdf, val_gaussavg = val(train_dataset[val_indices], lat_vecs[val_indices],model, optimizer, epoch, args)
            updatevalloss = True

            #if epoch >= 3:
            all_loss_train.append(train_loss)
            all_loss_val.append(val_loss)
            np_all_loss_train = np.ma.masked_where(np.array(all_loss_train) >= 2e9, np.array(all_loss_train))
            np_all_loss_val = np.ma.masked_where(np.array(all_loss_val) >= 2e9, np.array(all_loss_val))

            #if epoch >= 3:
            if epoch % 3 == 0:
                all_gaussavg_train.append(train_gaussavg)
                all_gaussavg_val.append(val_gaussavg)
                plotloss(outfolder, epoch, args.save_file_name, np_all_loss_train, np_all_loss_val)
                plotgauss(outfolder, epoch, args.save_file_name, all_gaussavg_train, all_gaussavg_val)

            is_best = abs(val_loss) < best_loss
            loss_epoch += 1
            test_dataset = SdfDataset(phase='test', args=args)
            getSurfacePoints(test_dataset, model, 100000, args)


            if updatevalloss and is_best:
                loss_epoch = 0
                best_loss = val_loss
                best_epoch = epoch
                best_points = mcube_points
                numhighloss = 0
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
                    curr_lr = param_group['lr']
                save_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, is_best, checkpoint_folder=args.checkpoint_folder)

            if epoch % 5 == 0:
                save_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, is_best, checkpoint_folder=args.checkpoint_folder)

            if args.scheduler == 'reducelr': 
                scheduler.step(val_loss)
            elif args.scheduler =='cosine':
                scheduler.step()

            save_curr_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, checkpoint_folder=args.checkpoint_folder)
            if epoch % 10 == 0:
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
            print(f"Epoch{epoch:d}. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}. Best Epoch: {best_epoch:d}. Best val loss: {best_loss:.8f}.",flush=True)

            print("Train_gauss = ",train_gaussavg)    
            print("Val_gauss = ",val_gaussavg)    
    if args.reg == 1:
        base_mcube_points, base_surf_area, base_mcube_sdf = getSurfacePoints(test_dataset, model, len(train_dataset)*2*args.train_batch, args)
        print("number of mcube points" , len(base_mcube_points))
        print("base surf area = ", base_surf_area)

        mcube_points = base_mcube_points # getSurfacePoints(model, len(train_dataset)*4*args.train_batch, 0, args)
        mcube_sdf = base_mcube_sdf
        #best_loss, val_indexcount = val(val_dataset, model, optimizer, -1, args, mcube_points)
        comparea = base_surf_area *0.2
        best_loss = 2e20
        n_mcube_points = len(mcube_points)
        print("=> Number of points in mcube points : %d" % n_mcube_points)
        mcube_sdf = base_mcube_sdf.copy()
        mcube_points = base_mcube_points.copy()
        for epoch in range(args.start_epoch, args.epochs):
            #train_indexcount = 1
            #val_indexcount = 1
          
            if args.mcube:
                train_loss, train_maxsdf, train_gaussavg = train(train_dataset, model, optimizer, epoch, args, mcube_points, mcube_sdf)
            else:
                train_loss, train_maxsdf, train_gaussavg = train(train_dataset, model, optimizer, epoch, args)
            updatevalloss = True

            #if epoch >= 3:
            all_loss_train.append(train_loss)
            np_all_loss_train = np.ma.masked_where(np.array(all_loss_train) >= 2e9, np.array(all_loss_train))

            plotloss(outfolder, epoch, args.save_file_name, np_all_loss_train)
            if epoch >= 3:
                all_gaussavg_train.append(train_gaussavg)
                plotgauss(outfolder, epoch, args.save_file_name, all_gaussavg_train)

            is_best = abs(train_loss) < best_loss
            loss_epoch += 1

            if args.resamp==2 and args.mcube:
                newmcube_points, newmcube_surf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, len(train_dataset)*2*args.train_batch, args)
                if not newmcube_points is None:
                    print("updating points ", len(newmcube_points))
                    mcube_points = newmcube_points.copy()
                    mcube_sdf = newmcube_sdf.copy()
                #n_mcube_points = len(mcube_points)
                #mcube_points_train = int(args.train_split_ratio * n_mcube_points)
                #mcube_points_train = int(0.05 * n_mcube_points)
                #full_indices = np.arange(n_mcube_points)
                #np.random.shuffle(full_indices)
                #train_mcube_indices = full_indices[:mcube_points_train]
                #val_mcube_indices = full_indices[mcube_points_train:]
                #print("train points = ", len(train_mcube_indices))
                #print("val points = ", len(val_mcube_indices))

            if args.resamp==3 and args.mcube and is_best:
                newmcube_points, newmcube_surf_area, newmcube_sdf = getSurfacePoints(test_dataset, model, len(train_dataset)*args.train_batch, args)
                if not newmcube_points is None:
                    mcube_points = newmcube_points.copy()
                    mcube_sdf = newmcube_sdf.copy()

            if is_best:
                #updateSamples(epoch, model, test_dataset, args)
                newmcube_points, updatevalloss = updateSamples(epoch, model, test_dataset, base_mcube_points, base_surf_area, is_best, args)
                #if not newmcube_points is None:
                #    mcube_points = newmcube_points

            if updatevalloss and is_best:
                loss_epoch = 0
                best_loss = train_loss
                best_epoch = epoch
                numhighloss = 0
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
                    curr_lr = param_group['lr']
                save_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, is_best, checkpoint_folder=args.checkpoint_folder)

            if epoch % 5 == 0:
                save_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, is_best, checkpoint_folder=args.checkpoint_folder)

            if args.scheduler == 'reducelr': 
                scheduler.step(train_loss)
            elif args.scheduler =='cosine':
                scheduler.step()

            save_curr_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()}, checkpoint_folder=args.checkpoint_folder)
            if epoch % 10 == 0:
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
            print(f"Epoch{epoch:d}. train_loss: {train_loss:.8f}. Best Epoch: {best_epoch:d}. Best val loss: {best_loss:.8f}.",flush=True)

            print("Train_gauss = ",train_gaussavg)    


