#import open3d
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from src.tools.initialize import initOptimizer, initlatcodeandmodelOptimizer, initlatentOptimizer, initScheduler, initDeepsdfDataSet
from src.tools.mcubeutils import plotImplicitCurvatureFromSamples, plotImplicitCurvature
from src.models.modeldeepsdf import Model
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
        #exit()
        self.initialize()
        self.initializeval(args.latlr)

        self.isreg1 = 0
        if args.resume:
            self.load_checkpoint()
        if args.reg:
            self.load_pretrained_checkpoint()
           # self.model.args.reg = 0
           # self.args.reg = 0
           # self.isreg1 = 1

    def prepare_data(self):
        os.makedirs(self.args.checkpoint_folder, exist_ok=True)

        self.train_dataset, self.val_dataset = initDeepsdfDataSet(self.args)
        if self.args.test:
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

        self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size = self.batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=generator)

        self.train_iter = iter(self.train_dataloader)
        self.val_iter = iter(self.val_dataloader)

    def initializetest(self, latlr):
        self.val_latent = torch.nn.Embedding(1, self.args.latcode, max_norm=1.0)   
        torch.nn.init.normal_(
            self.val_latent.weight.data,
            0.0,
            (1.0) / math.sqrt(self.args.latcode),
        )
        self.val_optimizer = initlatentOptimizer(self.val_latent, latlr)
        self.val_scheduler = initScheduler(self.val_optimizer, self.args)
        self.model.set_val_latent(self.val_latent)

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
        self.train_latent = torch.nn.Embedding(len(self.train_dataset), self.args.latcode, max_norm=1.0)   
        torch.nn.init.normal_(
            self.train_latent.weight.data,
            0.0,
            (1.0) / math.sqrt(self.args.latcode),
        )
        self.model = Model(self.train_latent, self.device, self.args).to(self.device)
        self.optimizer = initOptimizer(self.model.net, self.model.latent, self.args)
        self.scheduler = initScheduler(self.optimizer, self.args)
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
            #if 'best_train_loss' in checkpoint:
            #    self.best_train_loss = checkpoint['best_train_loss']
            if 'epoch' in checkpoint:
                if self.args.reg == 0:
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

    def run_test(self, test_index):
        #self.model.eval() 
        self.model.to(device)
        self.model.train()

      
        iters_every_epoch = int(int(float(self.args.totalsamples)/self.args.subsampple))/ self.batch_size)
        isbest = False
        for epoch in range(100):
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
                loss, gauss = self.model.run(batch, isTrain=False)
                if gauss is not None:
                    epochgauss.append(gauss)
                epochloss += batch_size * loss.item()
                count += batch_size
                loss.backward()
                self.val_optimizer.step()
                self.global_step += 1
                if self.global_step % 50 == 0:
                    logger.info(f'Loss = {self.global_step} {epochloss/count}')

            epochloss /= count
            self.val_scheduler.step(epochloss)
            is_best_loss = epochloss < self.best_val_loss
            if is_best_loss:
                self.best_val_loss = epochloss
                isbest = True
                print("validation Loss = {} {} {}".format(train_epoch, epochloss, train_latlr))
                plotImplicitCurvatureFromSamples(self.grid_512_samples, self.model, self.args, resolution=512, latent=self.val_latent)
                self.save_checkpoint('best_val_loss.tar')
            self.save_checkpoint('last_val_loss.tar')

            self.epoch += 1

    def run_val(self, train_epoch, train_latlr):
        #self.model.eval() 
        self.model.to(device)
        self.model.train()

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
                loss, gauss = self.model.run(batch, isTrain=False)
                if gauss is not None:
                    epochgauss.append(gauss)
                epochloss += batch_size * loss.item()
                count += batch_size
                loss.backward()
                self.val_optimizer.step()
                self.global_step += 1
                if self.global_step % 50 == 0:
                    logger.info(f'Loss = {self.global_step} {epochloss/count}')

            epochloss /= count
            self.val_scheduler.step(epochloss)
            is_best_loss = epochloss < self.best_val_loss
            if is_best_loss:
                self.best_val_loss = epochloss
                isbest = True
                print("validation Loss = {} {} {}".format(train_epoch, epochloss, train_latlr))
                plotImplicitCurvatureFromSamples(self.grid_512_samples, self.model, self.args, resolution=512, latent=self.val_latent)
                self.save_checkpoint('best_val_loss.tar')
            self.save_checkpoint('last_val_loss.tar')

            self.epoch += 1


    def rungausssteps(self, iters_every_epoch, epoch, max_epochs):
        epochgauss = []
        for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{max_epochs}]"):
            try:
                batch = next(self.train_iter)
            except Exception as e:
                self.train_iter = iter(self.train_dataloader)
                batch = next(self.train_iter)
            #print("batch size =", batch['index'].shape, flush=True)
            #print(batch['index'], flush=True)

            loss, gauss = self.model.run(batch, True)
            epochgauss.append(gauss)
        epochgauss = torch.hstack(epochgauss)
        plotImplicitCurvature(epochgauss, self.args)

    def run(self):
        self.model.train()
        self.model.to(device)

        if self.args.test:
            run_val(
        #iters_every_epoch = int(self.args.totalsamples/self.args.subsample)
        if self.args.reg == 1:
            subsamp = 2048
        else:
            subsamp = 18022
        iters_every_epoch = int((len(self.train_dataset)*int(float(self.args.totalsamples)/subsamp))/ self.batch_size)
        iters_every_epoch_gauss = iters_every_epoch # int(self.args.totalsamples/subsamp)
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

                batch_size = batch['index'].shape[0]
                self.optimizer.zero_grad()
                loss, gauss = self.model.run(batch)
                if gauss is not None:
                    epochgauss.append(gauss)
                epochloss += loss.item() *batch_size
                count += batch_size
                loss.backward()
                self.optimizer.step()
                self.global_step += 1

                if self.global_step % 50 == 0:
                    logger.info(f'Loss = {self.global_step} {epochloss/count}')

            epochloss /= count
            self.scheduler.step(epochloss)


            is_best_loss = epochloss < self.best_train_loss

            if is_best_loss:
                self.best_train_loss = epochloss
                logger.info(f'Best train Loss = {epoch} {epochloss}')
                rlatent = torch.tensor(np.random.randint(0, len(self.train_dataset))).to(self.device)
                self.model.set_test_latent(rlatent)
                self.model.eval()
                for param_group in self.optimizer.param_groups:
                    self.curr_lr = param_group['lr']
                    logger.info(f'LR step {self.curr_lr}')
                if epoch > 1:
                    plotImplicitCurvatureFromSamples(self.grid_128_samples, self.model, self.args, resolution=128, latent=self.train_latent(rlatent))
                if epoch > 1000:
                    plotImplicitCurvatureFromSamples(self.grid_256_samples, self.model, self.args, resolution=256, latent=self.train_latent(rlatent))
                    computeGauss = True
                    self.args.subsample = subsamp
                    self.rungausssteps(iters_every_epoch_gauss, epoch, max_epochs)
                    self.args.subsample = subsample_nogauss
                    computeGauss = False
                if epoch > 5000:
                    plotImplicitCurvatureFromSamples(self.grid_512_samples, self.model, self.args, resolution=512, latent=self.train_latent(rlatent))
                #if epoch > 10000:
                #    plotImplicitCurvatureFromSamples(self.grid_1024_samples, self.model.net, self.args, resolution=1024)
                self.save_checkpoint('best_train_loss.tar')

            self.save_checkpoint('last_loss.tar')
            #if self.epoch >= 10:
            #isbest = self.run_val(epoch, curr_latlr)
            #if isbest:
            #    self.save_checkpoint('best_train_val_loss_'+str(epoch)+'.tar')
            self.epoch += 1
