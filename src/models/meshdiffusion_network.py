import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from .layers import ScaleShiftGNMLPLayerSimple, GNMLPLayerSimple, GNMLPLayerSimpleRelu, ScaleShiftMLPLayerSimple, MLPLayerSimple, MLPLayerSigmoid
from .embedding import sinusoidalembedding,binarytimeembedding,timeembedding

from .loss import ConditionShapeMLPLoss
import math
from src.utils.writeply import *
from src.utils.utils import *


class VarianceScheduleTestSampling(Module):
    def __init__(self, num_steps, beta_1, beta_T, eta=0,mode='linear'):
        super().__init__()
        assert mode in ('linear', 'cosine' )
        self.mode = mode
        print(self.mode)
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.num_steps = num_steps


        if self.mode == 'linear':
            betas = torch.linspace(self.beta_1, self.beta_T, steps=(self.num_steps))
            self.num_steps = len(betas)

        elif self.mode == 'cosine':
            s = 0.008
            warmupfrac = 1
            frac_steps = int(self.num_steps * warmupfrac)
            rem_steps = self.num_steps - frac_steps
            ft = [math.cos(((t/self.num_steps + s)/(1+s))*(math.pi/2))**2 for t in range(num_steps+1)]
            #ft = [math.cos(((t/frac_steps + s)/(1+s))*(math.pi/2))**2 for t in range(frac_steps+1)]
            alphabar = [(ft[t]/ft[0]) for t in range(frac_steps+1)]
            betas = np.zeros(self.num_steps)
            for i in range(1,frac_steps+1):
                betas[i-1] = min(1-(alphabar[i]/alphabar[i-1]), 0.999)
            #betas[frac_steps:] = [beta_T]*rem_steps
            self.num_steps = len(betas)

        self.num_steps = len(betas)
        print("num steps samp = ", self.num_steps)

        betas = np.array(betas, dtype=np.float32)
        assert((betas > 0).all() and (betas <=1).all())

        alphas = 1 - betas
        alpha_cumprod = np.cumprod(alphas)
        alpha_cumprod_prev = np.append(1., alpha_cumprod[:-1])
        sigma = eta*np.sqrt((((1-alpha_cumprod_prev)/(1-alpha_cumprod))*(1-(alpha_cumprod/alpha_cumprod_prev))))
        sigma = torch.tensor(sigma)
        alphas_cumprod_prev = torch.tensor(alpha_cumprod_prev)
        sqrt_alpha_cumprod = torch.tensor(np.sqrt(alpha_cumprod))
        sqrt_one_minus_alpha_cumprod = torch.tensor(np.sqrt(1.0 - alpha_cumprod))
        log_one_minus_alpha_cumprod = torch.tensor(np.log(1.0 - alpha_cumprod))
        sqrt_recip_alpha_cumprod = torch.tensor(np.sqrt(1.0/alpha_cumprod))
        sqrt_recip_minus_one_alpha_cumprod = torch.tensor(np.sqrt((1.0/alpha_cumprod) -1))
        sqrt_recip_one_minus_alpha_cumprod = np.sqrt(1.0/(1 - alpha_cumprod))

        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)))
        posterior_mean_coeff1 = torch.tensor(betas * np.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod))
        posterior_mean_coeff2 = torch.tensor((1.0 - alpha_cumprod_prev)*np.sqrt(alphas) / (1 - alpha_cumprod))
        posterior_mean_coeff3 = torch.tensor((betas * sqrt_recip_one_minus_alpha_cumprod))
        betas = torch.tensor(betas)
        alphas = torch.tensor(alphas)
        alphas_cumprod = torch.tensor(alpha_cumprod)
        posterior_variance = torch.tensor(posterior_variance)

        self.register_buffer('test_betas', betas)
        self.register_buffer('test_alphas', alphas)
        self.register_buffer('test_sigma', sigma)
        self.register_buffer('test_alphas_cumprod', alphas_cumprod)
        self.register_buffer('test_alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('test_sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('test_sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)
        self.register_buffer('test_log_one_minus_alpha_cumprod', log_one_minus_alpha_cumprod)
        self.register_buffer('test_sqrt_recip_alpha_cumprod', sqrt_recip_alpha_cumprod)
        self.register_buffer('test_sqrt_recip_minus_one_alpha_cumprod', sqrt_recip_minus_one_alpha_cumprod)

        self.register_buffer('test_posterior_variance', posterior_variance)
        self.register_buffer('test_posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('test_posterior_mean_coeff1', posterior_mean_coeff1)
        self.register_buffer('test_posterior_mean_coeff2', posterior_mean_coeff2)
        self.register_buffer('test_posterior_mean_coeff3', posterior_mean_coeff3)

class VarianceScheduleMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.num_steps = config.num_steps
        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T
        assert self.mode in ('linear', 'cosine' )

        if self.mode == 'linear':
            betas = torch.linspace(self.beta_1, self.beta_T, steps=(self.num_steps))
            self.num_steps = len(betas)

        elif self.mode == 'cosine':
            s = 0.008
            warmupfrac = 1
            frac_steps = int(self.num_steps * warmupfrac)
            rem_steps = self.num_steps - frac_steps
            ft = [math.cos(((t/self.num_steps + s)/(1+s))*(math.pi/2))**2 for t in range(self.num_steps+1)]
            #ft = [math.cos(((t/frac_steps + s)/(1+s))*(math.pi/2))**2 for t in range(frac_steps+1)]
            alphabar = [(ft[t]/ft[0]) for t in range(frac_steps+1)]
            betas = np.zeros(self.num_steps)
            for i in range(1,frac_steps+1):
                betas[i-1] = min(1-(alphabar[i]/alphabar[i-1]), 0.999)
            #betas[frac_steps:] = [beta_T]*rem_steps
            self.num_steps = len(betas)

        betas = np.array(betas, dtype=np.float32)
        assert((betas > 0).all() and (betas <=1).all())

        alphas = 1 - betas
        alpha_cumprod = np.cumprod(alphas)
        alpha_cumprod_prev = np.append(1., alpha_cumprod[:-1])

        alphas_cumprod_prev = torch.tensor(alpha_cumprod_prev)
        sqrt_alpha_cumprod = torch.tensor(np.sqrt(alpha_cumprod))
        sqrt_one_minus_alpha_cumprod = torch.tensor(np.sqrt(1.0 - alpha_cumprod))
        log_one_minus_alpha_cumprod = torch.tensor(np.log(1.0 - alpha_cumprod))
        sqrt_recip_alpha_cumprod = torch.tensor(np.sqrt(1.0/alpha_cumprod))
        sqrt_recip_minus_one_alpha_cumprod = torch.tensor(np.sqrt((1.0/alpha_cumprod) -1))
        sqrt_recip_one_minus_alpha_cumprod = np.sqrt(1.0/(1 - alpha_cumprod))


        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)))
        posterior_mean_coeff1 = torch.tensor(betas * np.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod))
        posterior_mean_coeff2 = torch.tensor((1.0 - alpha_cumprod_prev)*np.sqrt(alphas) / (1 - alpha_cumprod))
        posterior_mean_coeff3 = torch.tensor((betas * sqrt_recip_one_minus_alpha_cumprod))
        betas = torch.tensor(betas)
        alphas = torch.tensor(alphas)
        alphas_cumprod = torch.tensor(alpha_cumprod)
        posterior_variance = torch.tensor(posterior_variance)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)
        self.register_buffer('log_one_minus_alpha_cumprod', log_one_minus_alpha_cumprod)
        self.register_buffer('sqrt_recip_alpha_cumprod', sqrt_recip_alpha_cumprod)
        self.register_buffer('sqrt_recip_minus_one_alpha_cumprod', sqrt_recip_minus_one_alpha_cumprod)

        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coeff1', posterior_mean_coeff1)
        self.register_buffer('posterior_mean_coeff2', posterior_mean_coeff2)
        self.register_buffer('posterior_mean_coeff3', posterior_mean_coeff3)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(self.num_steps), batch_size)
        #ts[1] = 50
        #ts[10] = 500
        #ts[15] = 1000
        #ts[50] = 5000
        #ts[55] = 8000
        #ts[55] = 600
        #ts[batch_size-1] = self.num_steps-1
        return ts.tolist()

class RankMLPNet(Module):
    def __init__(self, config):
        super().__init__()
        self.shape_dim = config.shape_dim
        #self.ranklayer = MLPLayerSigmoid(self.shape_dim, 1) 
        self.ranklayers = ModuleList([
            MLPLayerSimple(self.shape_dim, 1024),
            MLPLayerSimple(1024, 128),
            MLPLayerSigmoid(128, 1)])
        #self.ranklayers = ModuleList([
        #    MLPLayerSimple(self.shape_dim, 1024),
        #    MLPLayerSimple(1024, 128),
        #    MLPLayerSimple(128, 1)])

    def forward(self, x):
        out = x
        for layer in self.ranklayers:
            out = layer(out)

        return out

class MeshMLPNet(Module):
    def __init__(self, config):
        super().__init__()
        self.context_dim = config.context_dim
        self.time_dim = config.time_dim
        self.shape_dim = config.shape_dim
        self.flame_dim = config.flame_dim
        self.arch = config.arch

        if self.arch == 'archv3':
            self.layers = ModuleList([
                    ScaleShiftMLPLayerSimple(self.shape_dim, 1024, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple(1024, 512, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple(512, 256, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple(256, 128, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple(128, 64, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple(64, 128, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple((2*128), 256, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple((2*256), 512, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple((2*512), 1024, self.context_dim, self.time_dim),
                    ScaleShiftMLPLayerSimple((2*1024), self.shape_dim, self.context_dim, self.time_dim, islast=True),
                ])
            self.outmodule = ScaleShiftMLPLayerSimple(self.shape_dim, self.shape_dim, self.context_dim, self.time_dim, islast=True)
            self.flamelayers = ModuleList([
                    MLPLayerSimple(self.shape_dim, self.flame_dim, islast=True),
                ])
            self.skip_layers = [5,6,7,8]

        if self.arch == 'decoderv2':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.shape_dim, 1024, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple(1024, 512, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple(512, 256, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple(256, 128, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple(128, 64, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple(64, 128, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple((2*128), 256, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple((2*256), 512, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple((2*512), 1024, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple((2*1024), self.shape_dim, self.context_dim, self.time_dim, 8, islast=True),
                ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.shape_dim, self.shape_dim, None, None, 8, True)
            self.flamelayers = ModuleList([
                    GNMLPLayerSimple(self.shape_dim, self.flame_dim, islast=True),
                ])
            self.skip_layers = [5,6,7,8]

        if self.arch == 'decoderv7':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.shape_dim, 512, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple(512, 300, self.context_dim, self.time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(300, 512, self.context_dim, self.time_dim, 8),
                    ScaleShiftGNMLPLayerSimple((2*512), self.shape_dim, self.context_dim, self.time_dim, 8, islast=True),
                ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.shape_dim, self.shape_dim, None, None, 8, True)
            self.flamelayers = ModuleList([
                    GNMLPLayerSimple(self.shape_dim, self.flame_dim, islast=True),
                ])
            self.skip_layers = [2]

    def forward(self, x, t, context):
        """
        Args:
            x:  Mesh parameter at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        time_emb = sinusoidalembedding(t, self.time_dim)

        x = x.view(batch_size, -1)
        unet_out = []
        out = x.clone()
        k = 1
        for i, layer in enumerate(self.layers):
            out = layer(ctx=context, time=time_emb, x=out)

            if i < len(self.layers) - 1:
                unet_out.append(out.clone())
                if i in self.skip_layers:
                    out = torch.cat([out, unet_out[i-2*k]],dim=1)
                    k += 1

        out = out + self.outmodule(ctx=context, time=time_emb, x=x)
        flameout = out.clone()
        for i, layer in enumerate(self.flamelayers):
            flameout = layer(x=flameout)
        return out.view(batch_size, -1, 3), flameout.view(batch_size, -1)

class MeshDiffusion(Module):
    def __init__(self, net, var_sched:VarianceScheduleMLP, device, tag, with100):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.tag = tag
        self.device = device
        self.with100 = with100

    def decode(self, epoch, mesh_x0, context, getmeshx0=False): 
        """
        Args:
            mesh_x0:  Input flame parameters, (B, N, d) ==> Batch_size X Number of points X point_dim(3).
            context:  Image latent, (B, F). ==> Batch_size X Image_latent_dim 
            lossparam: NetworkLossParam object.
        """
        batch_size, _, _ = mesh_x0.size()

        t = None
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        #mesh_xT1 = torch.randn([batch_size, 5023, 3]).to(self.device)
        #trimesh.Trimesh(vertices=mesh_xT1[0].detach().cpu().numpy()).export('mesh_xT1.ply')
        if self.with100:
            mesh_xt, e_rand = self.get_train_mesh_sample(mesh_x0*100, t, epoch)
        else:
            mesh_xt, e_rand = self.get_train_mesh_sample(mesh_x0, t, epoch)
        predmesh_x0 = None
        #trimesh.Trimesh(vertices=(mesh_xt[1]).detach().cpu().numpy()).export('meshxt_1.ply')
        #trimesh.Trimesh(vertices=(mesh_xt[10]).detach().cpu().numpy()).export('meshxt_10.ply')
        #trimesh.Trimesh(vertices=(mesh_xt[15]).detach().cpu().numpy()).export('meshxt_15.ply')
        #trimesh.Trimesh(vertices=(mesh_xt[50]).detach().cpu().numpy()).export('meshxt_50.ply')
        #trimesh.Trimesh(vertices=(mesh_xt[55]).detach().cpu().numpy()).export('meshxt_55.ply')
        #trimesh.Trimesh(vertices=mesh_xt[batch_size-1].detach().cpu().numpy()).export('meshxt_bs.ply')
        #exit()

        e_theta, predmesh_x0, pred_flameparam  = self.get_network_prediction(mesh_xt=mesh_xt, t=t, context=context, prednoise=True, getmeshx0=True)
        #visualize_kde(t, mesh_x0*100, mesh_xt, predmesh_x0, "test", 0)
        
        #trimesh.Trimesh(vertices=predmesh_x0[15].detach().cpu().numpy()/100).export('meshx01_15.ply')
        #trimesh.Trimesh(vertices=predmesh_x0[25].detach().cpu().numpy()).export('meshx01_25.ply')
        #trimesh.Trimesh(vertices=predmesh_x0[50].detach().cpu().numpy()).export('meshx01_50.ply')
        #trimesh.Trimesh(vertices=predmesh_x0[55].detach().cpu().numpy()).export('meshx01_55.ply')
        #trimesh.Trimesh(vertices=predmesh_x0[batch_size-1].detach().cpu().numpy()).export('meshx01_bs.ply')
        if self.with100:
            return e_theta.view(batch_size,-1), e_rand.view(batch_size, -1), predmesh_x0/100, pred_flameparam
        return e_theta.view(batch_size,-1), e_rand.view(batch_size, -1), predmesh_x0, pred_flameparam
   
    def get_meshx0_from_noisepred(self, mesh_xt, e_theta, t):
        sqrt_recip_alpha_cumprod = self.var_sched.sqrt_recip_alpha_cumprod[t].view(-1,1,1)
        sqrt_recip_minus_one_alpha_cumprod = self.var_sched.sqrt_recip_minus_one_alpha_cumprod[t].view(-1,1,1)
        mesh_x0 =  (sqrt_recip_alpha_cumprod * mesh_xt) - (sqrt_recip_minus_one_alpha_cumprod * e_theta)
        return mesh_x0

    def get_meshx0_from_noisepred_sampling(self, mesh_xt, e_theta, t, varsched):
        t = torch.Tensor(t).long().to(self.device)
        sqrt_recip_alpha_cumprod = varsched.test_sqrt_recip_alpha_cumprod[t].view(-1,1,1).to(self.device)
        sqrt_recip_minus_one_alpha_cumprod = varsched.test_sqrt_recip_minus_one_alpha_cumprod[t].view(-1,1,1).to(self.device)
        mesh_x0 =  (sqrt_recip_alpha_cumprod * mesh_xt) - (sqrt_recip_minus_one_alpha_cumprod * e_theta)
        return mesh_x0

    def get_network_prediction(self, mesh_xt, t, context=None, prednoise=True, getmeshx0=False, issampling=False, varsched=None):
        mesh_xt = mesh_xt.to(dtype=torch.float32).to(self.device)
        t = torch.Tensor(t).long().to(self.device)
        if context is not None:
            context = context.to(self.device)
        pred_theta, pred_flameparam = self.net(mesh_xt, t=t, context=context)

        mesh_x0 = None
        if prednoise:
            e_theta = pred_theta
            if getmeshx0:
                if issampling and (varsched is not None):
                    mesh_x0 = self.get_meshx0_from_noisepred_sampling(mesh_xt, e_theta, t, varsched)
                else:
                    mesh_x0 = self.get_meshx0_from_noisepred(mesh_xt, e_theta, t)
        else:
            mesh_x0 = pred_theta
            e_theta = None
        return e_theta, mesh_x0, pred_flameparam

    def get_train_mesh_sample(self, mesh_x0, t, epoch=None):
        e_rand = torch.zeros(mesh_x0.shape).to(self.device)  # (B, N, d)
        batch_size = mesh_x0.shape[0]
        e_rand = torch.randn_like(mesh_x0)
        sqrt_alpha_cumprod = self.var_sched.sqrt_alpha_cumprod[t].view(-1,1,1)
        sqrt_one_minus_alpha_cumprod = self.var_sched.sqrt_one_minus_alpha_cumprod[t].view(-1,1,1)
        mesh_xt = (sqrt_alpha_cumprod * mesh_x0) + (sqrt_one_minus_alpha_cumprod * e_rand)
        return mesh_xt, e_rand 

    def get_shapemlp_loss(self, epoch, mesh_x0, context): # lossparam):
        """
        Args:
            mesh_x0:  Input point cloud, (B, N, d) ==> Batch_size X Number of points X point_dim(3).
            context:  Image latent, (B, F). ==> Batch_size X Image_latent_dim 
            lossparam: NetworkLossParam object.
        """
        batch_size, num_points, point_dim = mesh_x0.size()

        t = None
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        mesh_xt, e_rand = self.get_train_mesh_sample(mesh_x0, t, epoch)
        getmesh_x0 = False
        predmesh_x0 = None
        if (epoch >= 1) and (np.random.rand() < 0.001):
            getmesh_x0 = True
        e_theta, predmesh_x0, pred_flameparam = self.get_network_prediction(mesh_xt, t, context, True, getmesh_x0)

        if getmesh_x0:
            indx = np.random.randint(batch_size, size=(15,))
            sampt = np.array(t)[indx]
#            if context is None: 
#                radius = torch.Tensor([lossparam.mesh_radius] * batch_size).to(self.device)
#                center = torch.Tensor([lossparam.mesh_center] * batch_size).to(self.device)
#            else:
#                radius = lossparam.mesh_radius[indx]
#                center = lossparam.mesh_center[indx]
            #visualize_kde(sampt, mesh_x0, mesh_xt, predmesh_x0, self.tag, epoch)
            #visualize_prediction(sampt, predmesh_x0[indx], mesh_x0[indx], radius, center, 1.5, self.tag, epoch)
            convertToPLY(predmesh_x0[0].clone().detach().cpu().numpy(), str(epoch)+'_'+str(t[0])+'_'+self.tag+'_train_out.ply', '../results/')
            convertToPLY(mesh_x0[0].detach().cpu().numpy(), str(epoch)+'_'+ str(t[0])+'_'+self.tag+'_train_ref.ply', '../results/')

        loss = F.mse_loss(e_theta.view(-1, 3), e_rand.view(-1,3), reduction='mean')
        return loss

    def get_pposterior_sample(self, pred_mesh_x0, mesh_xt, t, varsched):
        posterior_mean_coeff1 = varsched.test_posterior_mean_coeff1[t].view(-1,1,1).to(self.device)
        posterior_mean_coeff2 = varsched.test_posterior_mean_coeff2[t].view(-1,1,1).to(self.device)
        posterior_variance = varsched.test_posterior_variance[t].view(-1,1,1).to(self.device)
        posterior_log_variance_clipped = varsched.test_posterior_log_variance_clipped[t].view(-1,1,1).to(self.device)
        mean = posterior_mean_coeff1 * pred_mesh_x0 + posterior_mean_coeff2 * mesh_xt
        return mean, posterior_log_variance_clipped, posterior_variance

    def get_pposterior_sample1(self, pred_flameparam_x0, e_theta, t, varsched):
        mean_coeff = torch.sqrt(varsched.test_alphas_cumprod_prev[t]).view(-1,1,1).to(self.device)
        dir_xt = torch.sqrt(1- varsched.test_alphas_cumprod_prev[t] - (varsched.test_sigma[t] **2)).view(-1,1,1).to(self.device)
        mean = mean_coeff * pred_flameparam_x0  + dir_xt * e_theta
        return mean

    def get_mean_var(self, mesh_xt, e_theta, t, varsched):
        posterior_mean_coeff3 = varsched.test_posterior_mean_coeff3[t].view(-1,1,1).to(self.device)
        sqrt_recip_alphas = torch.sqrt(1.0/ varsched.test_alphas[t]).view(-1,1,1).to(self.device)
        posterior_variance = torch.sqrt(varsched.test_posterior_variance[t]).view(-1,1,1).to(self.device)
        posterior_log_variance_clipped = ((0.5 * varsched.test_posterior_log_variance_clipped[t]).exp()).view(-1,1,1).to(self.device)
        c1 = ((1 - varsched.test_alphas[t])/(torch.sqrt(1 - varsched.test_alphas_cumprod[t]))).view(-1,1,1).to(self.device)
        #mean = sqrt_recip_alphas * (mesh_xt - posterior_mean_coeff3 * e_theta)
        mean = sqrt_recip_alphas * (mesh_xt - c1 * e_theta)
        return mean, posterior_variance, posterior_log_variance_clipped


    def sampletaubin(self, num_points, context, flame, batch_size=1, point_dim=3, sampling='ddim'): 
        mesh_xT = torch.randn([batch_size, num_points, 3]).to(self.device)
        context = context.to(self.device)
        varsched = VarianceScheduleTestSampling(self.var_sched.num_steps, 1e-4, 0.02, 'linear').to(self.device)
        plt.close()
        fig = plt.figure(figsize=(20,20))
        fig.suptitle('inference_prediction')
        iteri = 1

        r = np.random.randint(0, batch_size)
        print("number testing samples = ", varsched.num_steps)

        iterator = [x for x in reversed(range(0,varsched.num_steps))]

        traj = {varsched.num_steps-1: mesh_xT}
        #traj = {999: mesh_xT}
        for idx, t in enumerate(iterator):
            z = torch.zeros(mesh_xT.shape).to(self.device)  # (B, N, d)
            if t > 0:
                 z = torch.normal(0,1, size=(mesh_xT.shape)).to(self.device)

            mesh_xt = traj[t]
            batch_t = ([t]*batch_size)

            e_theta, predmesh_x0, pred_flameparam = self.get_network_prediction(mesh_xt, batch_t, context, getmeshx0=True, issampling=True, varsched=varsched)
            traj[-1] = e_theta

            newpredmesh = predmesh_x0.clone()
            for i in range(batch_size):
                faces = flame.faces_tensor.cpu()
                trimesh_m = trimesh.Trimesh(vertices=predmesh_x0[i].cpu().numpy(), faces=np.asarray(faces))
                trimesh_m = trimesh.smoothing.filter_taubin(trimesh_m, iterations=30)
                newpredmesh[i] = torch.Tensor(np.asarray(trimesh_m.vertices)).to(self.device)
            mesh_xt = newpredmesh
            #break
            if sampling == 'ddim':
               pred_flame_mesh_x0,_,_= flame(shape_params=pred_flameparam.float()) 
               mean, logvar, var = self.get_pposterior_sample(pred_flame_mesh_x0, mesh_xt, batch_t, varsched)
               mesh_xprevt = (mean + (0.5 * logvar).exp() * z)
            else:
               mean, var, logvar = self.get_mean_var(mesh_xt, e_theta, batch_t, varsched)
               mesh_xprevt = mean + logvar * z
            if t > 0:
               #print(iterator[idx+1])
               traj[iterator[idx+1]] = mesh_xprevt.clone().detach()     # Stop gradient and save trajectory.
               #traj[t-1] = mesh_xprevt.clone().detach()     # Stop gradient and save trajectory.
               del traj[t]
            else:
                traj[-1] = mesh_xprevt.clone().detach()
            #del traj[t]
        #e_theta, predmesh_x0, pred_flameparam = self.get_network_prediction(mesh_xt, batch_t, context, getmeshx0=True, issampling=True, varsched=varsched)
        
        return traj[-1], predmesh_x0, pred_flameparam


    def sample(self, num_points, context, flame, batch_size=1, point_dim=3, with_exp=False,sampling='ddim', shapeparam=None, expparam=None, jawparam=None): 
        mesh_xT = torch.randn([batch_size, num_points, 3]).to(self.device)
        context = context.to(self.device)
        #print(sampling)
        print(self.var_sched.beta_1)
        print(self.var_sched.beta_T)
        #exit()
        varsched = VarianceScheduleTestSampling(self.var_sched.num_steps, self.var_sched.beta_1, self.var_sched.beta_T, 0, self.var_sched.mode).to(self.device)

        iteri = 1
        faces = flame.faces_tensor.cpu()

        iterator = [x for x in reversed(range(0,varsched.num_steps))]
        #iterator = [x for x in reversed(range(-1,varsched.num_steps,10))]
        count = 0
        traj = {varsched.num_steps-1: mesh_xT}
        #traj = {999: mesh_xT}
        for idx, t in enumerate(iterator):
            z = torch.zeros(mesh_xT.shape).to(self.device)  # (B, N, d)
            if t > 0:
                 #z = torch.clamp(torch.randn(mesh_xT.shape), -1,1).to(self.device)
                 z = torch.normal(0,1, size=(mesh_xT.shape)).to(self.device)
                 z1 = torch.normal(0,0.01, size=(mesh_xT.shape)).to(self.device)

            mesh_xt = traj[t]
            batch_t = ([t]*batch_size)
            #print(sampling, flush=True)

            e_theta, predmesh_x0, pred_flameparam = self.get_network_prediction(mesh_xt, batch_t, context, getmeshx0=True, issampling=True, varsched=varsched)
            if sampling == 'dd':
                if shapeparam is not None:
                    pred_flameparam[:,:300] = shapeparam
                if expparam is not None:
                    pred_flameparam[:,300:400] = expparam
                return traj[t], predmesh_x0, pred_flameparam

            shape_param = pred_flameparam[:, :300]
            if self.net.flame_dim in [303,304]:
                rot_param = pred_flameparam[:,300:403]
            if self.net.flame_dim in [403, 406, 407]:
                exp_param, jaw_param = pred_flameparam[:,300:400], pred_flameparam[:,400:403]
            if self.net.flame_dim in [406, 407]:
                rot_param = pred_flameparam[:,403:406]
            if torch.isnan(e_theta).any():
                print(e_theta[0][0])
                print("index =", str(idx), flush=True)
                exit()
            #print(e_theta[0])

            if sampling == 'ddim1':
               if with_exp:
                   pred_flame_mesh_x0,_,_= flame(shape_params=shape_param.float(), expression_params=exp_param.float(), jaw_params=jaw_param.float())
               else:
                   pred_flame_mesh_x0,_,_= flame(shape_params=shape_param.float()) 
               center = (torch.max(pred_flame_mesh_x0, 1)[0] + torch.min(pred_flame_mesh_x0, 1)[0])/2.0
               pred_flame_mesh_x0 = (pred_flame_mesh_x0 - center.view(-1,1,3)).float().to(self.device)
               #mesh_xt = torch.clamp(mesh_xt, -0.5, 0.5)
               mean, logvar, var = self.get_pposterior_sample(pred_flame_mesh_x0, mesh_xt, batch_t, varsched)
               mesh_xprevt = (mean + (0.5 * logvar).exp() * z)

            elif sampling == 'ddim2':
               print("ddim2")
               if with_exp:
                   pred_flame_mesh_x0,_,_= flame(shape_params=shape_param.float(), expression_params=exp_param.float(), jaw_params=jaw_param.float())
               else:
                   pred_flame_mesh_x0,_,_= flame(shape_params=shape_param.float()) 
               center = (torch.max(pred_flame_mesh_x0, 1)[0] + torch.min(pred_flame_mesh_x0, 1)[0])/2.0
               pred_flame_mesh_x0 = (pred_flame_mesh_x0 - center.view(-1,1,3)).float().to(self.device)

               mean = self.get_pposterior_sample1(pred_flame_mesh_x0, e_theta, batch_t, varsched)
               mesh_xprevt = mean + varsched.test_sigma[t].view(-1,1,1) * z

            elif sampling == 'ddpm':
               mesh_xt = torch.clamp(mesh_xt, -0.5,0.5)
               #mesh_xt = torch.clamp(mesh_xt, -0.2, 0.2)
               mean, var, logvar = self.get_mean_var(mesh_xt, e_theta, batch_t, varsched)
               mesh_xprevt = (mean + logvar * z)
               predmesh_x0 = torch.clamp(predmesh_x0, -0.5, 0.5)
               #if t%10 == 0:
               #trimesh_m = trimesh.Trimesh(vertices=mesh_xprevt.squeeze().cpu().numpy(), faces=np.asarray(faces), process=False)
               #trimesh_m = trimesh.smoothing.filter_taubin(trimesh_m, iterations=30)
               #newpredmesh = torch.Tensor(np.asarray(trimesh_m.vertices)).to(self.device)
               #mesh_xprevt = newpredmesh.unsqueeze(0) 
               #print(predmesh_x0)
               #mesh_xprevt = torch.clamp(mesh_xprevt, -1,1)

            count += 1

            if t > 0:
               #print(iterator[idx+1])
               traj[iterator[idx+1]] = mesh_xprevt.clone().detach()     # Stop gradient and save trajectory.
               #traj[t-1] = mesh_xprevt.clone().detach()     # Stop gradient and save trajectory.
               del traj[t]
            else:
                traj[-1] = mesh_xprevt.clone().detach()
            #del traj[t]
        if shapeparam is not None:
            pred_flameparam[:,:300] = shapeparam
        #print(rank_param, flush=True)
        if self.with100:
            return traj[-1], predmesh_x0/100, pred_flameparam
        else:
            return traj[-1], predmesh_x0, pred_flameparam

