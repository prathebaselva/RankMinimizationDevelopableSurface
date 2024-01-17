#import open3d
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from src.tools.initialize import initOptimizer, initlatentOptimizer, initScheduler, initModel, initDeepsdfDataSet
from src.tools.mcube import getmcubePoints
from src.models.model import Model
from src.datasets.dataset import gridData
from tqdm import tqdm
import torch
import trimesh
import os
import math
import random
import numpy as np
from loguru import logger

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

        #if not os.path.isdir(args.checkpoint_folder):
        #    print("Creating new checkpoint folder " + args.checkpoint_folder)
        cudnn.benchmark = True
        self.prepare_data()
        self.initialize()
        self.initializeval(args.latlr)

        if args.resume:
            self.load_checkpoint()

    def prepare_data(self):
        #self.train_dataset, self.val_dataset = initDeepsdfDataSet(args)
        os.makedirs(self.args.checkpoint_folder, exist_ok=True)
        self.train_dataset, self.val_dataset = initDeepsdfDataSet(self.args)
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
        self.optimizer = initOptimizer(self.model.net, self.train_latent, self.args)
        self.scheduler = initScheduler(self.optimizer, self.args)
        self.grid_samples = gridData(self.args)

    def save_checkpoint(self, filename):
        model_dict = self.model.model_dict()
        model_dict['optimizer'] = self.optimizer.state_dict()
        model_dict['val_optimizer'] = self.val_optimizer.state_dict()
        model_dict['scheduler'] = self.scheduler.state_dict()
        model_dict['epoch'] = self.epoch
        model_dict['best_train_loss'] = self.best_train_loss
        model_dict['best_val_loss'] = self.best_val_loss
        model_dict['global_step'] = self.global_step
        model_dict['batch_size'] = self.batch_size

        path = os.path.join(self.args.checkpoint_folder, self.args.save_file_path)
        os.makedirs(path, exist_ok=True)
        savefilename = os.path.join(path, filename)

        torch.save(model_dict, savefilename)
    
    def load_checkpoint(self):
        model_path = os.path.join(self.args.checkpoint_folder, self.args.resume_checkpoint)
        map_location = {'cuda:%d' %0: 'cuda:%d' % 0}

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'val_optimizer' in checkpoint:
                self.val_optimizer.load_state_dict(checkpoint['val_optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'best_train_loss' in checkpoint:
                self.best_loss = checkpoint['best_train_loss']
            if 'best_val_loss' in checkpoint:
                self.best_loss = checkpoint['best_val_loss']
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            logger.info(f'Training resumes from {model_path}')
            logger.info(f'Training resumes from {self.epoch}')
            logger.info(f'Training resumes from {self.best_train_loss}')
            logger.info(f'Training resumes from {self.best_val_loss}')
            logger.info(f'Training resumes from {self.global_step}')
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


    def run(self):
        self.model.train()
        self.model.to(device)
        iters_every_epoch = int(len(self.train_dataset)/ self.batch_size)
        max_epochs = self.args.epochs
        for epoch in range(self.epoch, self.args.epochs):
            count = 0
            epochloss = 0
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{max_epochs}]"):
                try:
                    batch = next(self.train_iter)
                except Exception as e:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)

                batch_size = batch['index'].shape[0]

                self.optimizer.zero_grad()
                loss = self.model.run(batch)
                epochloss += loss.item() *batch_size
                count += batch_size
                loss.backward()
                self.optimizer.step()
                self.global_step += 1

                if self.global_step % 50 == 0:
                    logger.info(f'Loss = {self.global_step} {epochloss/count}')

            epochloss /= count
            self.scheduler.step(epochloss)

            curr_latlr = self.optimizer.param_groups[1]['lr']

            is_best_loss = epochloss < self.best_train_loss
            if is_best_loss:
                self.best_train_loss = epochloss
                logger.info(f'Best train Loss = {epoch} {epochloss}')
                for param_group in self.optimizer.param_groups:
                    curr_lr = param_group['lr']
                    logger.info(f'LR step {curr_lr}')
                self.save_checkpoint('best_train_loss.tar')

            self.save_checkpoint('last_loss.tar')
            #if self.epoch >= 10:
            isbest = self.run_val(epoch, curr_latlr)
            if isbest:
                self.save_checkpoint('best_train_val_loss_'+str(epoch)+'.tar')
            self.epoch += 1
