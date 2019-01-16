import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

class encoder(nn.Module):
    def __init__(self, emb_dim=300, layers=[200, 100], latent_dim=50):
        super(encoder, self).__init__()
        
        self.enc = nn.Sequential()
        self.enc.add_module('input', nn.Linear(emb_dim, layers[0]))
        
        for i in range(1, len(layers)):
            self.enc.add_module('linear_%d' % i, nn.Linear(layers[i-1], layers[i]))
            self.enc.add_module('tanh_%d' % i, nn.Tanh())
        
        self.enc.add_module('linear_%d' % len(layers), nn.Linear(layers[-1], latent_dim))
        self.enc.add_module('tanh_%d' % len(layers), nn.Tanh())

    def forward(self, inputs):
        out = self.enc(inputs)

        return out

class decoder(nn.Module):
    def __init__(self, emb_dim=300, layers=[100, 200], latent_dim=50):
        super(decoder, self).__init__()
        
        self.dec = nn.Sequential()
        self.dec.add_module('input', nn.Linear(latent_dim, layers[0]))
        
        for i in range(1, len(layers)):
            self.dec.add_module('linear_%d' % i, nn.Linear(layers[i-1], layers[i]))
            self.dec.add_module('tanh_%d' % i, nn.Tanh())

        self.dec.add_module('linear_%d' % len(layers), nn.Linear(layers[-1], emb_dim))
        self.dec.add_module('tanh_%d' % len(layers), nn.Tanh())

    def forward(self, inputs):
        out = self.dec(inputs)

        return out