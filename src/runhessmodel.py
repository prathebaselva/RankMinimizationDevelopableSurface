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


def getnewmcubepoints(validate_dataset, latent, model, epoch, iterloss, best_data, args):

        is_best_discmean = False 
        is_best_discmedian = False
        is_best_impmean = False
        is_best_impmedian = False
        is_best_loss = False
        mcube_points = None
        newmcube_gaussmean = [1000, 1000]
        newmcube_gaussmedian = [1000, 1000]
        discmean, discmedian, impmean, impmedian = 0, 0, 0, 0
        newmcube_points, newmcube_surf_area, newmcube_gaussmean, newmcube_gaussmedian, newmcube_sdf = getSurfacePoints(validate_dataset, latent, model, epoch, 500000, args, "resamp2")
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
        is_best_loss = iterloss < best_data[4]

        if is_best_discmean:
            print("Epoch {} has best disc mean".format(epoch))
            best_data[0] = discmean
            shutil.copy(os.path.join(outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(outfolder, args.save_file_name+'_bestdiscmean.obj'))
        if is_best_discmedian:
            print("Epoch  {} has best disc median".format(epoch))
            best_data[1] = discmedian
            shutil.copy(os.path.join(outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(outfolder, args.save_file_name+'_bestdiscmedian.obj'))
        if is_best_impmean:
            print("Epoch  {} has best imp mean".format(epoch))
            best_data[2] = impmean
            shutil.copy(os.path.join(outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(outfolder, args.save_file_name+'_bestimpmean.obj'))
        if is_best_impmedian:
            print("Epoch  {} has best imp median".format(epoch))
            best_data[3] = impmedian
            shutil.copy(os.path.join(outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(outfolder, args.save_file_name+'_bestimpmedian.obj'))
        if is_best_loss:
            print("Epoch  {} has best loss".format(epoch))
            best_data[4] = iterloss
            shutil.copy(os.path.join(outfolder, args.save_file_name+'_resamp2.obj'), os.path.join(outfolder, args.save_file_name+'_bestloss.obj'))

        return mcube_points, impmedian, best_data, np.array([is_best_discmean, is_best_discmedian, is_best_impmean, is_best_impmedian, is_best_loss])

#def runmodel(point_chunk, gtsdf_chunk, latent_chunk, model, optimizer, epoch, args, mcube_points=None, isTrain=True):
def testrunmodel(index, test_dataset, evaluate_dataset, latent, model, optimizer, scheduler, best_loss, args, mcube_points=None):
    #for i in range(num_batch):
    loss_sum = 0.0
    loss_count = 0.0
    regloss_sum = 0.0
    sdfloss_sum = 0.0
    best_loss = 2e10
    latpath = args.checkpoint_folder+'/../test_latent_abc7.pth.tar'
    startreg = 100
    if os.path.exists(latpath):
        latent = torch.load(latpath).to(device)
        best_latent = torch.load(latpath).to(device)
        startreg = 0
    else:
        best_latent = latent.clone()
        latent.requires_grad = True
    model.eval()
    mcube_points,_ , mcube_gaussvertmean, mcube_gaussvertmedian,_ = getSurfacePoints(evaluate_dataset, best_latent, model, -1, 300000, args)
    best_data = np.array([1000,1000,1000,1000,1000])
    extdropout = []
    for epoch in range(args.start_epoch,2000):
        if epoch == startreg:
            _,_, mcube_gaussvertmean, mcube_gaussvertmedian,_ = getSurfacePoints(evaluate_dataset, best_latent, model, epoch, 300001, args)
        loss_sum = 0.0
        loss_count = 0.0
        regloss_sum = 0.0
        sdfloss_sum = 0.0

        for chunk in range(550):
            fileindex, data = test_dataset[index]  # a dict
            optimizer.zero_grad()
            #print(data)
            sampled_points = data[:,0:3].to(device) # data['xyz'].to(device)
            #sampled_points = torch.tensor(point_chunk[i]).to(device) # data['xyz'].to(device)

            this_bs =  sampled_points.shape[0]
            lat_vec = latent.expand(this_bs,-1)
              
            sampled_points = torch.cat([lat_vec, sampled_points],dim=1).to(device)
            if args.useextdropout and epoch >  startreg:
                extdropout = [0,1,2,3,4]
            predicted_sdf = model(sampled_points, extdropout)
            #gt_sdf_tensor = torch.clamp(data['gt_sdf'].to(device), -args.clamping_distance, args.clamping_distance)
            gt_sdf_tensor = torch.clamp(data[:,3:].to(device), -args.clamping_distance, args.clamping_distance)


            sdfloss = datafidelity_testloss(predicted_sdf, gt_sdf_tensor, latent, epoch,  args)
            printinterval = 0
            if epoch % 3 == 0 and chunk == 1:
                print("data fidelity loss = ",sdfloss, flush=True)
                printinterval = 1
            if args.reg and args.mcube and epoch > startreg:
                mcube_r = np.random.randint( len(mcube_points), size=(this_bs,)) 
                mcube_sampled_points =  torch.tensor(mcube_points[mcube_r]).to(device)
                this_bs =  mcube_sampled_points.shape[0]
                mcube_sampled_points.requires_grad = True
                lat_mcube_sampled_points = torch.cat([lat_vec, mcube_sampled_points],dim=1).to(device)
                mcube_predicted_sdf = model(lat_mcube_sampled_points.float(),extdropout)
                hess_indices = np.arange(this_bs) # torch.where((torch.abs(mcube_predicted_sdf) <= args.threshold))[0]
                surfaceP_gradient,surfaceP_hessian_matrix  = getGradientAndHessian(mcube_predicted_sdf, mcube_sampled_points)
                regloss = implicit_loss(epoch, printinterval, surfaceP_gradient, surfaceP_hessian_matrix, None , args, hess_indices, mcube_predicted_sdf)
                #if printinterval:
                if printinterval:
                    print("regularizer loss = ",regloss)
                loss = sdfloss + regloss
            else:
                loss = sdfloss

            loss.backward()
            loss_sum += loss.item() * this_bs
            loss_count += this_bs
            #if epoch < 1000:
            optimizer.step()
            #print(loss_sum / loss_count, flush=True)
            
        iterloss = loss_sum / loss_count
        if args.reg and args.mcube and epoch > startreg:
            mcube_points,mcube_impmedian, best_data, best_bool = getnewmcubepoints(evaluate_dataset, latent, model, epoch,iterloss,best_data, args) 
            if best_bool.any():
                loss_epoch = 0
                best_epoch = epoch
                numhighloss = 0
                best_latent = latent.clone()
                for param_group in optimizer.param_groups: 
                    print("LR step :: ",param_group['lr']) 
                    curr_lr = param_group['lr']
                save_checkpoint({"epoch": epoch ,"test_latent":latent, "lr": curr_lr, "state_dict": model.state_dict(), "best_discmean": best_data[0], "best_discmedian": best_data[1], "best_impmean": best_data[2], "best_impmedian": best_data[3], "best_loss": best_data[4], "optimizer": optimizer.state_dict()}, best_bool[0], best_bool[1], best_bool[2], best_bool[3], best_bool[4], checkpoint_folder=args.checkpoint_folder)

            #scheduler.step(mcube_impmedian)
            scheduler.step(iterloss)

            save_curr_checkpoint({"epoch": epoch ,"test_latent":latent, "lr": curr_lr, "state_dict": model.state_dict(), "best_discmean": best_data[0], "best_discmedian": best_data[1], "best_impmean": best_data[2], "best_impmedian": best_data[3], "best_loss": best_data[4], "optimizer": optimizer.state_dict()}, checkpoint_folder=args.checkpoint_folder)
          
        elif args.reg == 0 or (args.reg and epoch <= startreg):
            if iterloss < best_loss: 
                print("iter  = ",epoch)
                print("best loss = ", iterloss, flush=True)
                print(latent[0:10], flush=True)
                for param_group in optimizer.param_groups:
                    print("LR step :: ",param_group['lr'])
                    curr_lr = param_group['lr']
                best_loss = iterloss
                best_latent = latent.clone()
                #_,_, mcube_gaussvertmean, mcube_gaussvertmedian,_ = getSurfacePoints(evaluate_dataset, latent, model, epoch, 300000, args)
                save_reg0checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict(), "test_latent":latent}, True, checkpoint_folder=args.checkpoint_folder)
            #latent = latent + torch.ones(args.lat).normal_(mean=0, std=0.01).to(device)
            scheduler.step(iterloss)
    _,_, mcube_gaussvertmean, mcube_gaussvertmedian,_ = getSurfacePoints(evaluate_dataset, best_latent, model, epoch, 300000, args)
    return best_loss, latent 


def runhessmodel(dataset, latent, model, optimizer, epoch, args, mcube_points=None, mcube_index=None, isTrain=True):
    loss_sum = 0.0
    loss_count = 0.0
    regloss_sum = 0.0
    sdfloss_sum = 0.0
    data_pos_sum = 0
    data_neg_sum = 0
    mcube_pos_sum = 0
    mcube_neg_sum = 0
    mcube_zero_sum = 0
    #num_batch = len(point_chunk)
    num_batch = len(mcube_index)
    #print("num_batch = ", num_batch)
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
    pinterval = np.random.randint(0,max(1,num_batch-1))

    #for i in range(num_batch):
    #for chunk in range(1500):
    #numiter = 100*num_batch #if isTrain else 110*num_batch
    numiter = 100 #1500
    for chunk in range(numiter):
     r = mcube_index
     #random.shuffle(r)
     for i in r:
        fileindex, data = dataset[i]  # a dict
        optimizer.zero_grad()
        #print(data)
        sampled_points = data[:,0:3].to(device) # data['xyz'].to(device)
        #sampled_points = torch.tensor(point_chunk[i]).to(device) # data['xyz'].to(device)

        this_bs =  sampled_points.shape[0]
        k = interval and (i == pinterval)

        lat_vec = latent(torch.tensor(i)).to(device)
     
        lat_vec = lat_vec.expand(this_bs,-1)
        sampled_points = torch.cat([lat_vec, sampled_points],dim=1).to(device)
        if not isTrain:
            with torch.no_grad():
                predicted_sdf = model(sampled_points, isTrain)
        else:
            predicted_sdf = model(sampled_points, isTrain)
            #gt_sdf_tensor = torch.clamp(data['gt_sdf'].to(device), -args.clamping_distance, args.clamping_distance)
        gt_sdf_tensor = torch.clamp(data[:,3:].to(device), -args.clamping_distance, args.clamping_distance)
        #gt_sdf_tensor = torch.tensor(gtsdf_chunk[i]).to(device) 
        if args.clamp:
            predicted_sdf = torch.clamp(predicted_sdf, -args.clamping_distance, args.clamping_distance)

        printinterval= 0
        if interval and (i == 0 or i == pinterval):
            printinterval = 1
        mcube_r = []
        if args.hessreg:
            if args.mcube:
                hess_indices = []
                valid_indices = []
                mcube_r = np.random.randint(len(mcube_points[i]), size=(this_bs,))
                mcube_sampled_points =  torch.tensor(mcube_points[i][mcube_r]).to(device)
                #surfaceP_points.append(mcube_sampled_points.clone().detach().cpu().numpy())
                #mcube_sampled_sdf = torch.tensor(mcube_sdf[r]).to(device)
                this_bs =  mcube_sampled_points.shape[0]
                mcube_sampled_points.requires_grad = True
                lat_vec.requires_grad = True
                lat_mcube_sampled_points = torch.cat([lat_vec, mcube_sampled_points],dim=1).to(device)
                #model.eval()
                mcube_predicted_sdf = model(lat_mcube_sampled_points.float(),isTrain)
                #model.train()
                #mcube_predicted_sdf = -mcube_predicted_sdf
                #mcube_predicted_sdf1 = mcube_predicted_sdf.view(mcube_predicted_sdf.size(0))

                #mcube_pos_sum += len(torch.where(mcube_predicted_sdf1 > 0)[0])
                #mcube_neg_sum += len(torch.where(mcube_predicted_sdf1 < 0)[0])
                #mcube_zero_sum += len(torch.where(mcube_predicted_sdf1 == 0)[0])
                    #maxsdf = torch.max(torch.abs(mcube_predicted_sdf))
                #pred_sdf = torch.abs(mcube_predicted_sdf1)
                #valid_indices = torch.where(pred_sdf <= args.threshold)[0]
                #hess_indices = torch.where((torch.abs(mcube_predicted_sdf) <= args.threshold))[0]
                hess_indices = np.arange(this_bs) # torch.where((torch.abs(mcube_predicted_sdf) <= args.threshold))[0]
                if len(hess_indices) <= 0:
                    continue

                surfaceP_gradient,surfaceP_hessian_matrix  = getGradientAndHessian(mcube_predicted_sdf, mcube_sampled_points)
                
                regloss = implicit_loss(epoch, printinterval, surfaceP_gradient, surfaceP_hessian_matrix, None , args, hess_indices, mcube_predicted_sdf)
                #base_mcube_hessian = getHessianforwnnm(model, base_mcube_points)
                    
                if printinterval:
                    print("regularizer loss = ",regloss)
           
            sdfloss = datafidelity_loss(predicted_sdf, gt_sdf_tensor, lat_vec, epoch,  args)
            #sdfloss, numpos, numneg = datafidelity_lossnormal(predicted_sdf, predicted_gradient, gt_sdf_tensor, sampled_normals , surfaceP_indices, args)
            if printinterval:
                print("data fidelity loss = ",sdfloss, flush=True)
            loss = regloss + sdfloss
        else:
            loss = datafidelity_loss(predicted_sdf, gt_sdf_tensor, lat_vec, epoch, args)
            #predicted_gradient = getGradient(predicted_sdf, sampled_points)
            #print(predicted_gradient[0:10])
            #loss, numpos, numneg = datafidelity_lossnormal(predicted_sdf, predicted_gradient, gt_sdf_tensor, sampled_normals , surfaceP_indices, args)

        if isTrain:
            #with torch.autograd.detect_anomaly():
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#            if printinterval:
#                total_norm = 0
#                for name,p in model.named_parameters():
#                    if p.requires_grad:
#                        print("name = ", name)
#                        param_norm = p.grad.data.norm(2)
#                        print(param_norm)
#                        total_norm += param_norm.item() ** 2
#                total_norm = total_norm **(1./2)
#                print("total norm = ", total_norm)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip, norm_type=2)
      
        loss_sum += loss.item() * this_bs
        #data_pos_sum += numpos
        #data_neg_sum += numneg
            
        loss_count += this_bs
        if isTrain:
            optimizer.step()

    if loss_count == 0:
        return 2e20, -512
    #print("data numpos = {}, numneg = {}".format(data_pos_sum, data_neg_sum))
    #print("mcube numpos = {}, numneg = {}, numzero={}".format(mcube_pos_sum, mcube_neg_sum, mcube_zero_sum))
    #print("mcube maxsdf = {}, minsdf = {}".format(maxsdf, minsdf))
    #args.data_delta = 1e4
    #args.mcube = 1
    return loss_sum, loss_count



