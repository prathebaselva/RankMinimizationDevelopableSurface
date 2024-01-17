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
    def __init__(self, in_features, out_features, bias=True, is_first=False, is_final=False,omega_0=30, norm='batch' ):
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.norm = norm
        self.is_final = is_final
        if self.norm == 'weight' and not self.is_final:
            self.linear = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=bias))
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    def getLinearLayer(self):
        return self.linear

    def forward(self, input):
        return torch.sin(self.omega_0 *self.linear(input))
  
    def forward_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class SDF(nn.Module):
    def __init__(
        self,
        channels,
        omega=30,
        latent_in= [],
        dropout=False,
        dropout_prob=0.2,
        norm='batch',
        final_linear=False,
        final_tanh=False,
    ):
        super(SDF, self).__init__()

        channels = [3] + channels + [1]

        self.num_layers = len(channels)
        self.latent_in = latent_in
        self.sine_layers = len(channels)
        self.omega_0 = omega
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.norm = norm
        self.final_tanh = final_tanh
        self.final_linear = final_linear
        
        #for layer in self.latent_in:
        #   channels[layer] = channels[layer] - channels[0]
        #linearlayer = nn.Linear(channels[-2], channels[-1], bias=True)
        #self.final_tanh_layer = nn.tanh(linearlayer)

        self.first_sine_layer = SineLayer(channels[0], channels[1], bias=True, is_first=True, is_final=False, omega_0=omega, norm=norm)
        self.intermediate_sine_layers = [SineLayer(channels[i], channels[i+1], bias=True, is_first=False, is_final=False, omega_0=omega, norm=norm) for i in range(1,self.num_layers-2)]
        self.final_sine_layer = SineLayer(channels[-2], channels[-1], bias=True, is_first=False, is_final=True, omega_0=omega, norm=norm)
        self.sine_net = [self.first_sine_layer] + self.intermediate_sine_layers + [self.final_sine_layer]
        self.sine_net = nn.Sequential(*self.sine_net)

        if norm == 'batch':
            self.batchnorm = []
            for i in range(len(channels)-2):        
                self.batchnorm.append(nn.BatchNorm1d(channels[i+1]))
            self.batchnorm = nn.Sequential(*self.batchnorm) 


    def forward(self, input, istrain=True):
        #print(len(input))
        d = input.ndim
        s = input.shape
        #print(s) 
        if d > 1:
            if s[1] > 3:
                x = input[:, -3:].float()
            else:
                x = input.float()
        else:
            if len(input) > 3:
                x = input[-3:].float()
            else:
                x = input.float()
        #x.register_hook(lambda grad: print(grad))
        #print(x.shape)
        for idx, layer in enumerate(self.sine_net):
            if layer in self.latent_in:
                x = torch.cat([x, input.float()],1)
            #print(layer)
            x = layer(x)
            if idx < self.num_layers - 2:  # except last layer
                if self.norm == 'batch':
                    x = self.batchnorm[idx](x)
                if self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=istrain)
        return x

def printgrad(y):
        print("input = ", torch.norm(y)) 
