import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .layers import GNMLPLayerSilu, GNMLPLayerTanh, GNMLPLayerElu, GNMLPLayerGelu, GNMLPLayerTan10h, MLPLayerSine


class MLPNet(nn.Module):
    def __init__(
        self,
        channels,
        arch='silu',
        dropout=0,
        dprob=0.5,
        latcode=256,
        omega=1,
        lasttanh=False,
    ):
        super(MLPNet, self).__init__()
        channels = [latcode + 3] + channels + [1]
        self.arch = arch

        self.num_layers = len(channels)
        if self.arch == 'sine':
            self.layers = nn.ModuleList([
                    MLPLayerSine(channels[0], channels[1], isfirst=True, omega=omega), 
                    MLPLayerSine(channels[1], channels[2], omega=omega), 
                    MLPLayerSine(channels[2], channels[3], omega=omega), 
                    MLPLayerSine(channels[3], channels[4], omega=omega), 
                    MLPLayerSine(channels[4]+3, channels[5], omega=omega), 
                    MLPLayerSine(channels[5], channels[6], omega=omega), 
                    MLPLayerSine(channels[6], channels[7], omega=omega), 
                    MLPLayerSine(channels[7], channels[8], omega=omega), 
                    MLPLayerSine(channels[8], channels[9], omega=omega, islast=True, istanh=lasttanh) 
            ])
        if self.arch == 'silu':
            self.layers = nn.ModuleList([
                    GNMLPLayerSilu(channels[0], channels[1]), 
                    GNMLPLayerSilu(channels[1], channels[2]), 
                    GNMLPLayerSilu(channels[2], channels[3]), 
                    GNMLPLayerSilu(channels[3], channels[4]), 
                    GNMLPLayerSilu(channels[4]+3, channels[5]), 
                    GNMLPLayerSilu(channels[5], channels[6]), 
                    GNMLPLayerSilu(channels[6], channels[7]), 
                    GNMLPLayerSilu(channels[7], channels[8]), 
                    GNMLPLayerSilu(channels[8], channels[9], islast=True, istanh=lasttanh) 
            ])
        if self.arch == 'elu':
            self.layers = nn.ModuleList([
                    GNMLPLayerElu(channels[0], channels[1]), 
                    GNMLPLayerElu(channels[1], channels[2]), 
                    GNMLPLayerElu(channels[2], channels[3]), 
                    GNMLPLayerElu(channels[3], channels[4]), 
                    GNMLPLayerElu(channels[4]+3, channels[5]), 
                    GNMLPLayerElu(channels[5], channels[6]), 
                    GNMLPLayerElu(channels[6], channels[7]), 
                    GNMLPLayerElu(channels[7], channels[8]), 
                    GNMLPLayerElu(channels[8], channels[9], islast=True) 
            ])
        if self.arch == 'gelu':
            self.layers = nn.ModuleList([
                    GNMLPLayerGelu(channels[0], channels[1], omega=omega), 
                    GNMLPLayerGelu(channels[1], channels[2], omega=omega), 
                    GNMLPLayerGelu(channels[2], channels[3], omega=omega), 
                    GNMLPLayerGelu(channels[3], channels[4], omega=omega), 
                    GNMLPLayerGelu(channels[4]+3, channels[5], omega=omega), 
                    GNMLPLayerGelu(channels[5], channels[6], omega=omega), 
                    GNMLPLayerGelu(channels[6], channels[7], omega=omega), 
                    GNMLPLayerGelu(channels[7], channels[8], omega=omega), 
                    GNMLPLayerGelu(channels[8], channels[9], omega=omega, islast=True) 
            ])
        if self.arch == 'tanh':
            self.layers = nn.ModuleList([
                    GNMLPLayerTanh(channels[0], channels[1]), 
                    GNMLPLayerTanh(channels[1], channels[2]), 
                    GNMLPLayerTanh(channels[2], channels[3]), 
                    GNMLPLayerTanh(channels[3], channels[4]), 
                    GNMLPLayerTanh(channels[4]+3, channels[5]), 
                    GNMLPLayerTanh(channels[5], channels[6]), 
                    GNMLPLayerTanh(channels[6], channels[7]), 
                    GNMLPLayerTanh(channels[7], channels[8]), 
                    GNMLPLayerTanh(channels[8], channels[9], islast=True) 
            ])
        self.skiplayer = [4] # [4]
        self.dropout = dropout
        self.dprob = dprob


    def forward(self,x):
        out = x.clone()
        #print(self.training)

        count = 0
        for layer in self.layers:
            out = layer(out)
            if self.dropout:
                out = F.dropout(out,p=0.5, training=self.training)
            if count+1 in self.skiplayer:
                out = torch.cat([out, x[:,-3:]], dim=1)
            count += 1
        return out

        #return {"xyz": x[:-3:], "output":out}
