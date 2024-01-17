import torch
import math
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


class MLPLayerSimple(Module):
    def __init__(self, dim_in, dim_out, islast=False):
        super(MLPLayerSimple, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(dim_out)
        self.islast = islast

    def forward(self, x, islast=False):
        if self.islast:
            return self.mlp(x)
        ret = self.relu(self.bn(self.mlp(x)))
        return ret

class MLPLayerSigmoid(Module):
    def __init__(self, dim_in, dim_out, islast=False):
        super(MLPLayerSigmoid, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.bn = torch.nn.BatchNorm1d(dim_out)

    def forward(self, x, islast=False):
        out = self.bn(self.mlp(x))
        #print(out[0:10])
        #print(out.shape)
        #out -= out.min(0, keepdim=True)[0]
        #out /= out.max(0, keepdim=True)[0]

        return torch.sigmoid(out)


class MLPLayerSine(Module):
    def __init__(self, dim_in, dim_out, groups=8, omega=30, isfirst=False, islast=False, istanh=False):
        super(MLPLayerSine, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.tanh = torch.nn.Tanh()
        self.istanh = istanh
        self.isfirst = isfirst
        self.omega = omega
        

    def init_weight(self):
        with torch.no_grad():
            if self.isfirst:
                a = np.sqrt(6/(dim_in + dim_out))
            else:
                a = (np.sqrt(6/(dim_in + dim_out))/self.omega)

        self.mlp.weight.uniform(-a,a)

    def forward(self, x):
        ret = torch.sin(self.omega*self.mlp(x))
        return ret

class GNMLPLayerSilu(Module):
    def __init__(self, dim_in, dim_out, groups=8, islast=False, istanh=False):
        super(GNMLPLayerSilu, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.silu = torch.nn.SiLU()
        self.tanh = torch.nn.Tanh()
        self.istanh = istanh
        
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)
        self.islast = islast

    def forward(self, x):
        if self.islast:
            if self.istanh:
                return (self.tanh(self.mlp(x)))
            return self.mlp(x)
        ret = self.gn(self.silu(self.mlp(x)))
        return ret

class GNMLPLayerTanh(Module):
    def __init__(self, dim_in, dim_out, groups=8, islast=False):
        super(GNMLPLayerTanh, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.tanh = torch.nn.Tanh()
        
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)
        self.islast = islast

    def forward(self, x):
        if self.islast:
            return (self.mlp(x))
        ret = self.gn(self.tanh(self.mlp(x)))
        return ret

class GNMLPLayerTan10h(Module):
    def __init__(self, dim_in, dim_out, groups=8, islast=False):
        super(GNMLPLayerTan10h, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.tanh = torch.nn.Tanh()
        
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)
        self.islast = islast

    def forward(self, x):
        if self.islast:
            return (self.mlp(x))
        ret = self.gn(self.tanh(10*self.mlp(x)))
        return ret

class GNMLPLayerElu(Module):
    def __init__(self, dim_in, dim_out, groups=8, islast=False):
        super(GNMLPLayerElu, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.elu = torch.nn.ELU()
        
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)
        self.islast = islast

    def forward(self, x):
        if self.islast:
            return (self.mlp(x))
        ret = self.gn(self.elu(self.mlp(x)))
        return ret
    
class GNMLPLayerGelu(Module):
    def __init__(self, dim_in, dim_out, groups=8, islast=False, istanh=False, omega=1):
        super(GNMLPLayerGelu, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.gelu = torch.nn.GELU()
        self.tanh = torch.nn.Tanh()
        self.istanh = istanh
        self.omega = omega
        
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)
        self.islast = islast

    def forward(self, x):
        if self.islast:
            if self.istanh:
                return (self.tanh(self.mlp(x)))
            return (self.mlp(x))
        ret = self.gn(self.gelu(self.omega*(self.mlp(x))))
        return ret


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def lr_func(epoch):
    if epoch <= start_epoch:
        return 1.0
    elif epoch <= end_epoch:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / start_lr)
    else:
        return end_lr / start_lr
