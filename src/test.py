#import open3d
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from src.tools.initialize import initlatcodeandmodelOptimizer, initlatentOptimizer, initScheduler, initDeepsdfTestDataSet
from src.tools.mcubeutils import plotImplicitCurvatureFromSamples
from src.models.modeldeepsdf import Model
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

class Tester(object):
    def __init__(self, device, args):
        self.device = device
        self.batch_size = args.batch_size
        self.best_test_loss = 2e20
        self.best_epoch = -1
        self.args = args
        self.epoch = 0
        self.global_step = 0
        self.lr = self.args.lr
        self.curr_lr = self.args.lr
        self.testindex = args.testindex
        self.opttype = args.opttype

        #if not os.path.isdir(args.checkpoint_folder):
        #    print("Creating new checkpoint folder " + args.checkpoint_folder)
        cudnn.benchmark = True
        self.prepare_data()
        self.initialize(args.latlr)
        if self.args.resume:
            self.load_pretrained_checkpoint()

    def prepare_data(self):
        os.makedirs(self.args.checkpoint_folder, exist_ok=True)
        self.test_dataset = initDeepsdfTestDataSet(self.args)
        generator = torch.Generator()
        generator.manual_seed(0)
        self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size = self.batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=generator)
        self.test_iter = iter(self.test_dataloader)
  
    def initialize(self, test_latlr):
        self.test_latent = torch.ones(self.args.latcode).normal_(mean=0, std=(1.0/math.sqrt(self.args.latcode))).to(self.device)
        #self.test_latent = torch.nn.Embedding(1, self.args.latcode, max_norm=1.0)   
        #torch.nn.init.normal_(
        #    self.test_latent.weight.data,
        #    0.0,
        #    (1.0) / math.sqrt(self.args.latcode),
        #)
        self.test_latent.requires_grad = True

        self.model = Model(self.test_latent, self.device, self.args).to(self.device)
        self.model.set_test_latent(self.test_latent)
        if self.opttype == 'lat':
            self.optimizer = initlatentOptimizer(self.test_latent, test_latlr)
        else:
            self.optimizer = initlatcodeandmodelOptimizer(self.model.net, self.test_latent, self.args)
        self.scheduler = initScheduler(self.optimizer, self.args)
        #print(self.model.net.parameters())
        self.grid_128_samples = gridData(grid_N=128)
        self.grid_256_samples = gridData(grid_N=256)
        self.grid_512_samples = gridData(grid_N=512)
        self.grid_1024_samples = gridData(grid_N=1024)

    def save_checkpoint(self, filename):
        model_dict = self.model.model_dict()
        model_dict['optimizer'] = self.optimizer.state_dict()
        model_dict['scheduler'] = self.scheduler.state_dict()
        model_dict['test_latent'] = self.best_latent
        model_dict['epoch'] = self.epoch
        model_dict['best_test_loss'] = self.best_test_loss
        model_dict['global_step'] = self.global_step
        model_dict['batch_size'] = self.batch_size
        model_dict['lr'] = self.curr_lr

        path = os.path.join(self.args.checkpoint_folder, self.args.save_file_path)
        os.makedirs(path, exist_ok=True)
        savefilename = os.path.join(path, filename)

        torch.save(model_dict, savefilename)
    
    def load_pretrained_checkpoint(self):
        model_path = os.path.join(self.args.pretrained_folder, self.args.resume_checkpoint)
        map_location = {'cuda:%d' %0: 'cuda:%d' % 0}

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'test_latent' in checkpoint:
                print('test_latent')
                self.test_latent = checkpoint['test_latent'].clone()
            else:
                print("test latent not found")
                exit()
            if 'optimizer' in checkpoint:
                print("optimizer")
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.optimizer.param_groups[0]['lr'] = self.args.lr
            if 'scheduler' in checkpoint:
                print("scheduler")
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'lr' in checkpoint:
                self.curr_lr = checkpoint['lr']
                self.lr = checkpoint['lr']
            if 'epoch' in checkpoint:
                if self.args.reg == 0:
                    self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            logger.info(f'Training resumes from pretrained {model_path}')
            logger.info(f'Training resumes from epoch : {self.epoch}')
            logger.info(f'Training resumes from lr : {self.curr_lr}')
        else:
            logger.info(f'{model_path} not found ')

    def run(self, test_index):
        self.model.to(device)
        self.model.train()
        isbest = False
        self.model.latent.requires_grad = True
        self.test_latent.requires_grad = True
        for epoch in range(100):
            loss = 0
            epochloss = 0
            count = 0
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{max_epochs}]"):
                try:
                    batch = next(self.test_iter)
                except Exception as e:
                    self.test_iter = iter(self.test_dataloader)
                    batch = next(self.test_iter)

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

    def run(self):
        self.model.to(device)
        self.model.training = False

        iters_every_epoch = int(int(float(self.args.totalsamples)/self.args.subsample))
        logger.info(iters_every_epoch) 
        max_epochs = self.args.epochs
        isbest = False
        #self.initialize(test_latlr)
        self.global_step = 0
        self.best_lat = 0
        self.model.latent.requires_grad = True
        if self.args.reg == 0:
            self.test_latent.requires_grad = True
        for epoch in range(self.epoch, self.args.epochs):
        #for epoch in range(self.epoch, 10):
            loss = 0
            epochloss = 0
            count = 0
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{max_epochs}]"):
                try:
                    batch = next(self.test_iter)
                except Exception as e:
                    self.test_iter = iter(self.test_dataloader)
                    batch = next(self.test_iter)

                batch_size = batch['index'].shape[0]
                self.optimizer.zero_grad()
                loss, gauss = self.model.run(batch)
                epochloss += loss.item()*batch_size
                count += batch_size
                loss.backward()
                self.optimizer.step()
                self.global_step += 1
            epochloss /= count
            logger.info(f'epochloss {epochloss}')

            self.scheduler.step(epochloss)
            is_best_loss = epochloss < self.best_test_loss
            if is_best_loss:
                self.best_test_loss = epochloss
                self.best_latent = self.test_latent.clone()
                #print(self.best_latent, flush=True)
                #print(self.test_latent)
                
                curr_lr = self.optimizer.param_groups[0]['lr']
                print("Test Loss = {} {} {}".format(epoch, epochloss, curr_lr), flush=True)
                #plotImplicitCurvatureFromSamples(self.grid_128_samples, self.model, self.args, resolution=128, latent=self.test_latent)
                #rotv, simplices = getmcubePoints(self.grid_128_samples, self.best_latent, self.model, self.args)
                #trimesh.Trimesh(np.array(rotv), np.array(simplices)).export(self.args.save_file_path+'_'+str(self.testindex)+'.obj')
                self.save_checkpoint('best_train_loss.tar')
                if self.args.reg:
                    if epoch <= 900:
                        plotImplicitCurvatureFromSamples(self.grid_128_samples, self.model, self.args, resolution=128, latent=self.test_latent, filename=self.args.save_file_path)
                    else:
                        plotImplicitCurvatureFromSamples(self.grid_512_samples, self.model, self.args, resolution=512, latent=self.test_latent, filename=self.args.save_file_path)
                else:
                    if epoch <= 500:
                        plotImplicitCurvatureFromSamples(self.grid_128_samples, self.model, self.args, resolution=128, latent=self.test_latent, filename=self.args.save_file_path)
                    else:
                        plotImplicitCurvatureFromSamples(self.grid_512_samples, self.model, self.args, resolution=512, latent=self.test_latent, filename=self.args.save_file_path)
            self.epoch += 1
            curr_lr = self.optimizer.param_groups[0]['lr']
            if curr_lr < 1e-08:
                print('Learning rate is 1e-08, early stopnning')
                break
        #rotv, simplices = getmcubePoints(self.grid_samples, self.best_latent, self.model.net, self.args)
        #trimesh.Trimesh(np.array(rotv), np.array(simplices)).export(self.args.save_file_path+'_'+str(self.testindex)+'.obj')
        #return isbest
