from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np

def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.encoder_decoder = nn.Sequential(
            # Input : nc x 32 x 32
#            nn.Conv2d(params['nc'], params['ngf'], 4, 2, 1, bias=False),
#            nn.ReLU(inplace=True),
            # ngf x 16 x 16
#            nn.Conv2d(params['ngf'], params['ngf']*2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(params['ngf']*2),
#            nn.ReLU(inplace=True),
            # (ngf x 2) x 8 x 8
#            nn.Conv2d(params['ngf']*2, params['ngf']*4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(params['ngf']*4),
#            nn.ReLU(inplace=True),
            # (ngf x 4) x 4 x 4
#            nn.Conv2d(params['ngf']*4, params['nz'], 4, bias=False),
#            nn.BatchNorm2d(params['nz']),
#            nn.ReLU(inplace=True),

            # nz
            nn.ConvTranspose2d(params['nz'], params['ngf']*4, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(params['ngf']*4),
            nn.ReLU(inplace=True),
            # (ngf x 4 x 4)
            nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(params['ngf']*2),
            nn.ReLU(inplace=True),
            # (ngf x 8 x 8)
            nn.ConvTranspose2d(params['ngf']*2, params['ngf'], 4, 2, 1, bias=False),
            nn.InstanceNorm2d(params['ngf']),
            nn.ReLU(inplace=True),
            # ngf x 16 x 16
            nn.ConvTranspose2d(params['ngf'], params['nc'], 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder_decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # nc x 32 x 32
            nn.Conv2d(params['nc'], params['ndf'], 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            # ndf x 16 x 16
            nn.Conv2d(params['ndf'], params['ndf']*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(params['ndf']*2),
            nn.ReLU(inplace=True),
            # (ndf x 2) x 8 x 8
            nn.Conv2d(params['ndf']*2, params['ndf']*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(params['ndf']*4),
            nn.ReLU(inplace=True),
            # (ndf x 4) x 4 x 4
            nn.Conv2d(params['ndf']*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)
