import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


device = torch.device("cpu")
deviceids = []
if torch.cuda.is_available():
    print("cude")
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
  
    def forward_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class MLP_GN(nn.Module):
    def __init__(self, channels, num_layers, isDecoder=False, bias = False):
        #super(MLP, self).__init(channels, num_layers, droput_prob=0.5, bias=False)
        super(MLP_GN, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        groupNorm = []
        if not isDecoder:
            for i in range(1,self.num_layers):
                if self.channels[i] >= 64:
                    groupNorm.append(self.channels[i]//64)
                else:
                    groupNorm.append(1)
        else:
            for i in range(1,self.num_layers-1):
                if self.channels[i] >= 64:
                    groupNorm.append(self.channels[i]//64)
                else:
                    groupNorm.append(1)
            groupNorm.append(1)
        self.mlpmodules = torch.nn.ModuleList([nn.Sequential(nn.Linear(self.channels[i], self.channels[i+1]), nn.LeakyReLU(negative_slope=0.2), nn.GroupNorm(groupNorm[i],self.channels[i+1])) for i in range(self.num_layers-1)])

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight.data)

    def forward(self, input_features):
        x = input_features
        for i in range(len(self.mlpmodules)):
            mlpmod = self.mlpmodules[i]
            x = mlpmod(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        channels,
        omega=30,
        dropout=None,
        dropout_prob=0.2,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=False
    ):
        super(Decoder, self).__init__()

        channels = [3] + channels + [1]

        self.num_layers = len(channels)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.use_tanh = use_tanh
        self.sine_layers = len(channels)
        self.weight_norm = weight_norm
        self.omega_0 = omega
        self.channels = channels

        groupNorm = []
        for i in range(1,self.num_layers-1):
            if channels[i] >= 64:
                groupNorm.append(channels[i]//64)
            else:
                groupNorm.append(1)
        groupNorm.append(1)

        #self.linear_modules = torch.nn.ModuleList([nn.Sequential(nn.Linear(channels[i], channels[i+1]), torch.sin(self.omega_0), nn.GroupNorm(groupNorm[i],self.channels[i+1])) for i in range(self.num_layers-1)])
        self.first_linear_module = nn.Sequential(SineLayer(channels[0], channels[1], True, True), nn.GroupNorm(groupNorm[0],self.channels[1]))
        self.linear_modules = [nn.Sequential(SineLayer(channels[i], channels[i+1]), nn.GroupNorm(groupNorm[i],self.channels[i+1])) for i in range(1,self.num_layers-1)]
        self.linear_modules = nn.ModuleList([self.first_linear_module]+self.linear_modules)
        self.first_initialize_weights()
        self.initialize_weights()


    def first_initialize_weights(self):
        for m in self.modules():
            #if isinstance(m, nn.Linear):
            #    m.weight.uniform_(-1/m.in_features, 1/m.in_features)
            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight.data)

    def initialize_weights(self):
        for m in self.modules():
            #if isinstance(m, nn.Linear):
            #    m.weight.uniform_(-np.sqrt(6/m.in_features/self.omega_0), np.sqrt(6 /m.in_features / self.omega_0))
            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight.data)

    def forward(self, input, istrain=True):
        #print(len(input))
        if len(input) > 1:
            x = input[:, -3:]
        else:
            x = input[-3:]
        
        #print(x)
        #if istrain:
        #    h = x.register_hook(printgrad)
    

        for lin in (self.linear_modules):
            #print(lin)
            #if layer in self.latent_in:
            #    x = torch.cat([x, input], 1)
            x = lin(x)
            #print(x.shape)
        #x = torch.sin(self.omega_0 * x)
        return x


def printgrad(y):
        print("input = ", torch.norm(y)) 
