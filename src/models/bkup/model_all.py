import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class ActivationLayerBeforeNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation='tanh', is_first=False, is_final=False, omega=30, withomega=True, norm='weight', init='def'):
        super(ActivationLayerBeforeNorm, self).__init__()
        self.omega = omega
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.is_final = is_final
        self.init = init
        self.withomega = withomega
        self.norm = norm
        if self.norm == 'weight' and not self.is_final:
            self.linear = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=bias))
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.activation = activation
        self.th = nn.Tanh()
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        if self.init == 'xav':
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # Xavier initialization
            if self.is_first:
                if self.activation == 'tanh':
                    a = (5/3)*(np.sqrt(6/(self.in_features + self.out_features)))
                elif self.activation == 'sineorg':
                    a = 1.0 / self.in_features
                else:
                    a = (np.sqrt(6/(self.in_features + self.out_features)))
            else:
                if self.activation == 'tanh':
                    if self.withomega:
                        a = (5/3)*(np.sqrt(6/(self.in_features + self.out_features))/self.omega)
                    else:
                        a = (5/3)*(np.sqrt(6/(self.in_features + self.out_features)))
                elif self.activation == 'sineorg':
                        a = (np.sqrt(6/(self.in_features))/self.omega)
                elif self.activation == 'sinh':
                    if self.withomega:
                        a = (np.sqrt(6/(self.in_features + self.out_features))/self.omega.sum())
                    else:
                        a = (np.sqrt(6/(self.in_features + self.out_features)))
                else:
                    if self.withomega:
                        a = (np.sqrt(6/(self.in_features + self.out_features))/self.omega)
                    else:
                        a = (np.sqrt(6/(self.in_features + self.out_features)))
                    #std = (np.sqrt(6/(self.in_features + self.out_features))/self.omega)
            self.linear.weight.uniform_(-a, a)
            nn.init.zeros_(self.linear.bias)
                

    def getLinearLayer(self):
        return self.linear

    def forward(self, x):
        y = self.linear(x)
        if self.activation == 'tanh':
            return self.th(self.omega*y)
        if self.activation == 'relu':
            return self.relu(y)
        elif self.activation == 'gelu':
            return self.gelu(self.omega* y)
        elif self.activation == 'silu':
            return self.silu(self.omega* y)
        elif self.activation == 'sinelu':
            return (y+torch.sin(self.omega*y))*torch.sigmoid(0.05*y)
        elif self.activation == 'sinh':
            k = []
            for o in self.omega:
                k.append(torch.sin(o*y))
            return torch.stack(k).sum(dim=0)/len(self.omega)
        elif (self.activation == 'sine' or self.activation == 'sineorg'):
            return torch.sin(self.omega*y)
  

class ActivationLayerAfterNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation='tanh', is_first=False, is_final=False,omega=30, withomega=True, init='def',norm='batch', groupfeature=None):
        super(ActivationLayerAfterNorm, self).__init__()
        self.omega = omega
        self.withomega = withomega
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.is_final = is_final
        self.init = init
        self.groupfeature = groupfeature
        self.norm = norm
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        self.norm = norm
        self.th = nn.Tanh()
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()

        if not self.is_final:
            if norm == 'layer':
                self.layer = nn.Sequential(self.linear, nn.LayerNorm(out_features))
            if norm == 'group':
                self.layer = nn.Sequential(self.linear, nn.GroupNorm(groupfeature, out_features))
            if norm == 'batch':
                self.layer = nn.Sequential(self.linear, nn.BatchNorm1d(out_features))
            if norm == 'weight':
                self.layer = nn.utils.weight_norm(self.linear)
            if norm == 'none':
                self.layer = self.linear
        else:
            self.layer = self.linear

        if self.init == 'xav':
            self.init_weights()


    def init_weights(self):
        with torch.no_grad():
            # Xavier initialization
            for m in self.modules():
                if isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight.data)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight.data)
                if isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight.data)
                if isinstance(m, nn.Linear):
                    if self.is_first:
                        if self.activation == 'tanh':
                            a = (5/3)*(np.sqrt(6/(self.in_features + self.out_features)))
                        elif self.activation == 'sineorg':
                            a = 1.0 / self.in_features
                        else:
                            a = (np.sqrt(6/(self.in_features + self.out_features)))
                    else:
                        if self.activation == 'tanh':
                            if self.withomega:
                                a = (5/3)*(np.sqrt(6/(self.in_features + self.out_features))/self.omega)
                            else:
                                a = (5/3)*(np.sqrt(6/(self.in_features + self.out_features)))
                        elif self.activation == 'sineorg':
                                a = (np.sqrt(6/(self.in_features))/self.omega)
                        elif self.activation == 'sinh':
                            if self.withomega:
                                a = (np.sqrt(6/(self.in_features + self.out_features))/self.omega.sum())
                            else:
                                a = (np.sqrt(6/(self.in_features + self.out_features)))
                        else:
                            if self.withomega:
                                a = (np.sqrt(6/(self.in_features + self.out_features))/self.omega)
                            else:
                                a = (np.sqrt(6/(self.in_features + self.out_features)))
                    #std = (np.sqrt(6/(self.in_features + self.out_features))/self.omega)
                    nn.init.uniform_(m.weight, -a, a)
                    nn.init.zeros_(m.bias)
                    #self.linear.weight.uniform_(-a, a)
                

    def getLinearLayer(self):
        return self.linear

    def forward(self, x):
        y = self.layer(x)
        if self.activation == 'tanh':
            return self.th(self.omega*y)
        elif self.activation == 'gelu':
            return self.gelu(self.omega* y)
        elif self.activation == 'silu':
            return self.silu(self.omega* y)
        elif self.activation == 'sinelu':
            return (y+torch.sin(self.omega*y))*torch.sigmoid(0.05*y)
        elif self.activation == 'sinh':
            k = []
            for o in self.omega:
                k.append(torch.sin(o*y))
            return torch.stack(k).sum(dim=0)/len(self.omega)
        elif self.activation == 'sine':
            return torch.sin(self.omega*y)
        elif self.activation == 'sineorg':
            return torch.sin(self.omega*y)
  

class Model(nn.Module):
    def __init__(
        self,
        channels,
        dropout=None,
        dropout_prob=0.2,
        norm_layers=(),
        latent_in=(),
        omega=1,
        withomega=True,
        activation='tanh',
        norm='layer',
        init='def',
        latcode=256,
        activationafternorm=True,
    ):
        super(Model, self).__init__()
        channels = [latcode + 3] + channels + [1]

        self.num_layers = len(channels)
        self.latent_in = latent_in
        self.latent_size = latcode
        self.num_layers = len(channels)
        self.omega = omega
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.norm = norm
        self.activation = activation
        
        groupNorm = []
        for i in range(1,self.num_layers-1):
            if channels[i] >= 64:
                groupNorm.append(channels[i]//64)
            else:
                groupNorm.append(1)
        groupNorm.append(1)

        #for i in range(1,self.num_layers-1):`
        #    if i in self.latent_in:
        #        channels[i] = channels[i] + channels[0]
        inchannel = []
        for i in range(self.num_layers):
            inlayer = channels[i]
            if i in self.latent_in:
                inlayer += channels[0]
            inchannel.append(inlayer)

        if norm == 'layer':
            self.first_layer = nn.Sequential(ActivationLayerBeforeNorm(inchannel[0], channels[1], True, activation, True, False, omega, withomega, norm, init), nn.LayerNorm(channels[1]))
            self.hidden_layers = [nn.Sequential(ActivationLayerBeforeNorm(inchannel[i], channels[i+1], True, activation, False, False, omega, withomega , norm, init), nn.LayerNorm(channels[i+1])) for i in range(1,self.num_layers-2)]
        if norm == 'group':
            self.first_layer = nn.Sequential(ActivationLayerBeforeNorm(inchannel[0], channels[1], True, activation, True, False, omega, withomega ,norm, init), nn.GroupNorm(groupNorm[0],channels[1]))
            self.hidden_layers = [nn.Sequential(ActivationLayerBeforeNorm(inchannel[i], channels[i+1], True ,activation, False, False, omega, withomega, norm, init), nn.GroupNorm(groupNorm[i],channels[i+1])) for i in range(1,self.num_layers-2)]
        if norm == 'batch':
            self.first_layer = nn.Sequential(ActivationLayerBeforeNorm(channels[0], channels[1], True, activation, True, False, omega, withomega ,norm,init), nn.BatchNorm1d(channels[1]))
            self.hidden_layers = [nn.Sequential(ActivationLayerBeforeNorm(channels[i], channels[i+1], True ,activation, False, False, omega, withomega ,norm,init), nn.BatchNorm1d(channels[i+1])) for i in range(1,self.num_layers-2)]
        if (norm == 'none' or norm =='weight'):
            self.first_layer = ActivationLayerBeforeNorm(inchannel[0], channels[1], True, activation, True, False, omega, withomega,norm, init)
            self.hidden_layers = [ActivationLayerBeforeNorm(inchannel[i], channels[i+1], True ,activation, False, False, omega, withomega,norm, init) for i in range(1,self.num_layers-2)]
#            self.hidden_layers = []
#            for i in range(1, self.num_layers-2):
#                inlayer = channels[i]
#                if i in self.latent_in:
#                    inlayer += channels[0]
#                self.hidden_layers.append(ActivationLayerBeforeNorm(inlayer, channels[i+1], True ,activation, False, False, omega, withomega,norm, init))
#        inlayer = channels[-2]
#        if self.num_layers-2 in self.latent_in:
#            inlayer += channels[0]
        if self.activation == 'relu':
            self.final_layer = ActivationLayerBeforeNorm(inchannel[-2], channels[-1], True, 'tanh', False, True, omega, withomega,norm, init)
        else:
            self.final_layer = ActivationLayerBeforeNorm(inchannel[-2], channels[-1], True, activation, False, True, omega, withomega,norm, init)
            #self.final_layer = ActivationLayerBeforeNorm(channels[-2]+channels[0], channels[-1], True, activation, False, True, omega, withomega,norm, init)

        self.net = nn.ModuleList([self.first_layer]+self.hidden_layers+[self.final_layer])
        if init == 'xav':
            self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight.data)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight.data)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight.data)

    # input: N x 3
    def forward(self, inputxyz):
        #print(len(input))
        #if len(input) > 1:
        #    x = input[:, -3:]
        #else:
        #    x = input[-3:]
        x = inputxyz.float()
        xyz = inputxyz.clone().float()
        for index, layer in enumerate(self.net):
            if index in self.latent_in:
                x = torch.cat([x, xyz], dim=1).float()
            x = layer(x)
            #zr = torch.zeros(x.size())
            if self.dropout is not None and index in self.dropout:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)

        return x


