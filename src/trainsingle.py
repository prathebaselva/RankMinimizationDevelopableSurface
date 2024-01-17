#import open3d
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from src.tools.initialize import initNoLatentOptimizer, initScheduler, initDeepsdfSingleDataSet
from src.tools.mcubeutils import plotImplicitCurvatureFromSamples, plotImplicitCurvature
from src.tools.mcube import getImplicitCurvatureforSamples
from src.tools.colorcode import getGaussColorcoded
from src.models.model import Model
from src.datasets.dataset import gridData
from tqdm import tqdm
import torch
import trimesh
import os
import math
import random
import numpy as np
import open3d as o3d
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt

outfolder = 'output/'

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)
   

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer(object):
    def __init__(self, device, args):
        self.device = device
        self.batch_size = args.batch_size
        self.best_train_loss = 2e20
        self.best_val_loss = 2e20
        self.best_epoch = -1
        self.args = args
        self.epoch = 0
        self.global_step = 0
        self.lr = self.args.lr
        self.curr_lr = self.args.lr

        #if not os.path.isdir(args.checkpoint_folder):
        #    print("Creating new checkpoint folder " + args.checkpoint_folder)
        cudnn.benchmark = True
        self.prepare_data()
        self.initialize()
        #self.initializeval(args.latlr)

        self.isreg1 = 0
        if args.resume:
            self.load_checkpoint()
        elif args.reg:
            self.load_pretrained_checkpoint()
           # self.model.args.reg = 0
           # self.args.reg = 0
           # self.isreg1 = 1

    def prepare_data(self):
        os.makedirs(self.args.checkpoint_folder, exist_ok=True)
        self.train_dataset, self.gtpoints = initDeepsdfSingleDataSet(self.args)
        self.gtpoints = torch.Tensor(self.gtpoints).float()
        generator = torch.Generator()
        generator.manual_seed(0)
        self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=generator)


        self.train_iter = iter(self.train_dataloader)
   
    def initializeval(self, train_latlr):
        self.val_latent = torch.nn.Embedding(len(self.val_dataset), self.args.latcode, max_norm=1.0)   
        torch.nn.init.normal_(
            self.val_latent.weight.data,
            0.0,
            (1.0) / math.sqrt(self.args.latcode),
        )
        self.val_optimizer = initlatentOptimizer(self.val_latent, train_latlr)
        self.val_scheduler = initScheduler(self.val_optimizer, self.args)
        self.model.set_val_latent(self.val_latent)

    def initialize(self):
        self.model = Model(self.device, self.args).to(self.device)
        self.optimizer = initNoLatentOptimizer(self.model.net, self.args)
        self.scheduler = initScheduler(self.optimizer, self.args)
        self.grid_64_samples = gridData(grid_N=64)
        self.grid_128_samples = gridData(grid_N=128)
        self.grid_256_samples = gridData(grid_N=256)
        self.grid_512_samples = gridData(grid_N=512)
        self.grid_1024_samples = gridData(grid_N=1024)

    def save_checkpoint(self, filename):
        model_dict = self.model.model_dict()
        model_dict['optimizer'] = self.optimizer.state_dict()
        #model_dict['val_optimizer'] = self.val_optimizer.state_dict()
        model_dict['scheduler'] = self.scheduler.state_dict()
        model_dict['epoch'] = self.epoch
        model_dict['best_train_loss'] = self.best_train_loss
        model_dict['best_val_loss'] = self.best_val_loss
        model_dict['global_step'] = self.global_step
        model_dict['batch_size'] = self.batch_size
        model_dict['lr'] = self.curr_lr

        path = os.path.join(self.args.checkpoint_folder, self.args.save_file_path)
        os.makedirs(path, exist_ok=True)
        savefilename = os.path.join(path, filename)

        torch.save(model_dict, savefilename)
    
    def load_checkpoint(self):
        #model_path = os.path.join(self.args.checkpoint_folder, self.args.resume_checkpoint)
        model_path = os.path.join(self.args.pretrained_folder, self.args.resume_checkpoint)
        map_location = {'cuda:%d' %0: 'cuda:%d' % 0}

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'best_train_loss' in checkpoint:
                self.best_train_loss = checkpoint['best_train_loss']
            if 'epoch' in checkpoint:
                #if self.args.reg == 0:
                self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            if 'lr' in checkpoint:
                self.curr_lr = checkpoint['lr']
            logger.info(f'Training resumes from model : {model_path}')
            logger.info(f'Training resumes from epoch : {self.epoch}')
            logger.info(f'Training resumes from lr : {self.curr_lr}')
            logger.info(f'Training resumes from best train loss : {self.best_train_loss}')
            logger.info(f'Training resumes from global step : {self.global_step}')
        else:
            logger.info(f'{model_path} not found ')

    def load_pretrained_checkpoint(self):
        model_path = os.path.join(self.args.pretrained_folder, self.args.resume_checkpoint)
        #model_path = os.path.join(self.args.resume_checkpoint)
        map_location = {'cuda:%d' %0: 'cuda:%d' % 0}

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'optimizer' in checkpoint:
                print("optimizer")
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.optimizer.param_groups[0]['lr'] = self.args.lr
            if 'scheduler' in checkpoint:
                print("scheduler")
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            #if 'lr' in checkpoint:
            #    self.curr_lr = checkpoint['lr']
            if 'epoch' in checkpoint:
                if self.args.reg == 0:
                    self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            logger.info(f'Training resumes from pretrained {model_path}')
            logger.info(f'Training resumes from epoch : {self.epoch}')
            logger.info(f'Training resumes from lr : {self.curr_lr}')
            logger.info(f'Training resumes from best train loss : {self.best_train_loss}')
        else:
            logger.info(f'{model_path} not found ')

    def run_val(self, train_epoch, train_latlr):
        #self.model.eval() 
        self.model.to(device)

        iters_every_epoch = int(len(self.val_dataset)/ self.batch_size)
        max_epochs = 1
        isbest = False
        self.initializeval(train_latlr)
        for epoch in range(1):
            loss = 0
            epochloss = 0
            count = 0
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{max_epochs}]"):
                try:
                    batch = next(self.val_iter)
                except Exception as e:
                    self.val_iter = iter(self.val_dataloader)
                    batch = next(self.val_iter)

                batch_size = batch['index'].shape[0]
                self.val_optimizer.zero_grad()
                loss = self.model.run(batch, isTrain=False)
                epochloss += batch_size * loss.item()
                count += batch_size
                loss.backward()
                self.val_optimizer.step()

            epochloss /= count
            self.val_scheduler.step(epochloss)
            is_best_loss = epochloss < self.best_val_loss
            if is_best_loss:
                self.best_val_loss = epochloss
                isbest = True
                print("validation Loss = {} {} {}".format(train_epoch, epochloss, train_latlr))
                rlatent = torch.tensor(np.random.randint(0, len(self.val_dataset))).to(self.device)
                rotv, simplices = getmcubePoints(self.grid_samples, self.val_latent(rlatent), self.model.net, self.args)
                trimesh.Trimesh(np.array(rotv), np.array(simplices)).export(self.args.save_file_path+'_val.obj')
                self.save_checkpoint('best_val_loss.tar')
                self.save_checkpoint('best_val_loss_'+str(train_epoch)+'.tar')
        return isbest

    def rungausssteps(self, iters_every_epoch, istest=False):
        epochgauss = []
        epochmincurv = []
        epochmaxcurv = []
        epochdet = []
        self.model.eval()
        plotImplicitCurvatureFromSamples(self.grid_512_samples, self.model, self.args, resolution=512)
        mesh = o3d.io.read_triangle_mesh(self.args.save_file_path+'_512.obj') 
        pcl = mesh.sample_points_poisson_disk(number_of_points=250000)
        points = np.asarray(pcl.points)
        print(points.shape, flush=True)
        iters_every_epoch = int(250000/self.args.subsample)
        start = 0
        for step in tqdm(range(iters_every_epoch+1)):
            end = np.min((start+int(self.args.subsample), 250000))
            #points.append(batch['data'][:,:,0:3].squeeze().cpu().numpy())
            #gauss, mean, pcurv = self.model.runtest(batch, True)
            gauss, mean, pcurv, det = self.model.runtestondata(points[start:end,], True)
            epochmincurv.append(pcurv[1])
            epochmaxcurv.append(pcurv[0])
            epochgauss.append(gauss)
            epochdet.append(det)
            #points = np.vstack(np.array(points))
            #print(points.shape)
            start = end
        epochgauss = torch.hstack(epochgauss)
        epochmincurv = torch.hstack(epochmincurv)
        epochmaxcurv = torch.hstack(epochmaxcurv)
        epochdet = torch.hstack(epochdet)
        #getGaussColorcoded(points[0:end,], epochgauss.squeeze().cpu().numpy(), 25, 'gauss', self.args.fname+'_gauss')
        #getGaussColorcoded(points[0:end,], epochmincurv.squeeze().cpu().numpy(), 5, 'mingauss', self.args.fname+'_mingauss')
        plotImplicitCurvature(epochgauss, epochmincurv, epochmaxcurv, epochdet, self.args)
        np.save(self.args.save_file_path+'_imppoints.npy', points)
        self.model.train()


    def run(self):
        self.model.train()
        self.model.to(device)
        if self.args.test:
            self.model.eval()
            print("in testing", flush=True)
            #plotImplicitCurvatureFromSamples(self.grid_1024_samples, self.model, self.args, resolution=1024)
            #plotImplicitCurvatureFromSamples(self.grid_64_samples, self.model, self.args, resolution=64)
            #plotImplicitCurvatureFromSamples(self.grid_128_samples, self.model, self.args, resolution=128)
            #plotImplicitCurvatureFromSamples(self.grid_256_samples, self.model, self.args, resolution=256)
            #plotImplicitCurvatureFromSamples(self.grid_512_samples, self.model, self.args, resolution=512)
            #exit()
            self.args.subsample = 2048
            iters_every_epoch = int(self.args.totalsamples/self.args.subsample)
            computeGauss = True
            self.rungausssteps(iters_every_epoch)
            #self.args.subsample = subsample_nogauss
            computeGauss = False
            exit()
        #exit()
        #self.gtpoints.to(device)
        #self.gtpoints.requires_grad = True
        #iters_every_epoch = int(len(self.train_dataset)/ self.batch_size)
        iters_every_epoch = int(self.args.totalsamples/self.args.subsample)
        if self.args.reg == 1:
            subsamp = 2048
        else:
            subsamp = 4096
        iters_every_epoch_gauss = int(self.args.totalsamples/subsamp)
        subsample_nogauss = self.args.subsample
        max_epochs = self.args.epochs
        for epoch in range(self.epoch, self.args.epochs):
            count = 0
            epochloss = 0
            epochgauss = []
            self.model.train()
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{max_epochs}]"):
                try:
                    batch = next(self.train_iter)
                except Exception as e:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                batch_size = batch['data'].shape[1]
                self.optimizer.zero_grad()
                loss, gauss = self.model.run(batch)# mcubesamples[mcuberandom])
                if gauss is not None:
                    epochgauss.append(gauss)
                epochloss += loss.item() *batch_size
                count += batch_size
                loss.backward()
                #if self.args.losstype == 'logdet':
                #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                self.optimizer.step()
                self.global_step += 1

                if self.global_step % 51 == 0:
                     logger.info(f'Loss = {self.global_step} {epochloss/count}')

            epochloss /= count
            self.scheduler.step(epochloss)

            is_best_loss = epochloss < self.best_train_loss

            if is_best_loss:
                self.best_train_loss = epochloss
                logger.info(f'Best train Loss = {epoch} {epochloss}')
                for param_group in self.optimizer.param_groups:
                    self.curr_lr = param_group['lr']
                    logger.info(f'LR step {self.curr_lr}')
                #self.model.eval()
                #with torch.no_grad():
                plotImplicitCurvatureFromSamples(self.grid_128_samples, self.model, self.args, resolution=128)
                #if epoch > 10:
                #    plotImplicitCurvatureFromSamples(self.grid_128_samples, self.model, self.args, resolution=128)
                if epoch > 500:
                    plotImplicitCurvatureFromSamples(self.grid_256_samples, self.model, self.args, resolution=256)
                    computeGauss = True
                    self.args.subsample = subsamp
                    self.rungausssteps(iters_every_epoch_gauss)
                    self.args.subsample = subsample_nogauss
                    computeGauss = False
                if epoch > 1000:
                    plotImplicitCurvatureFromSamples(self.grid_512_samples, self.model, self.args, resolution=512)
            #if epoch > 10000:
            #    plotImplicitCurvatureFromSamples(self.grid_1024_samples, self.model.net, self.args, resolution=1024)
                self.save_checkpoint('best_train_loss.tar')

            self.save_checkpoint('last_loss.tar')
            #self.model.train()
            #if self.epoch >= 10:
            #isbest = self.run_val(epoch, curr_latlr)
            #if isbest:
            #    self.save_checkpoint('best_train_val_loss_'+str(epoch)+'.tar')
            self.epoch += 1
