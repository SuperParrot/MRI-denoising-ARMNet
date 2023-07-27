import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio

from models.ResNet import ResNet


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class MultiDila_Block(nn.Module):
    def __init__(self, params):
        super(MultiDila_Block, self).__init__()

        C=params[0]

        self.cbamLayer=CBAMLayer(C)

        self.dil_conv1=nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=C//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            
            nn.Conv2d(in_channels=C//4, out_channels=C//4, kernel_size=(3, 1), stride=1, dilation=1, padding=(1, 0), bias=False),
            nn.Conv2d(in_channels=C//4, out_channels=C//4, kernel_size=(1, 3), stride=1, dilation=1, padding=(0, 1), bias=False),
        )
        
        self.dil_conv2=nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=C//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            
            nn.Conv2d(in_channels=C//4, out_channels=C//4, kernel_size=(3,1), stride=1, dilation=2, padding=(2,0), bias=False),
            nn.Conv2d(in_channels=C//4, out_channels=C//4, kernel_size=(1,3), stride=1, dilation=2, padding=(0,2), bias=False),
        )
        
        self.dil_conv3=nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=C//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            
            nn.Conv2d(in_channels=C//4, out_channels=C//4, kernel_size=(3,1), stride=1, dilation=3, padding=(3,0), bias=False),
            nn.Conv2d(in_channels=C//4, out_channels=C//4, kernel_size=(1,3), stride=1, dilation=3, padding=(0,3), bias=False),
        )

        self.pool=nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=C, out_channels=C//4, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.ln_nl=nn.Sequential(
            #nn.BatchNorm2d(C, affine=False),
            #nn.GroupNorm(max(int(C//4),1), C),
            nn.LayerNorm([64, 64]),
            nn.GELU(),
        )

    def forward(self,x):
        x = torch.cat((self.dil_conv1(x), self.dil_conv2(x), self.dil_conv3(x), self.pool(x)), dim=1)
        x = self.ln_nl(x)
        x = self.cbamLayer(x)

        return x


class MultiDilaResnet(nn.Module):
    def __init__(self, C_in, C_out, num_layers, features=64):
        super(MultiDilaResnet, self).__init__()

        self.backbone=nn.Sequential(
            nn.Conv2d(in_channels=C_in, out_channels=features, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GELU(),

            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GELU(),

            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )

        self.resnet=ResNet(num_layers, MultiDila_Block, features)

        self.end = nn.Conv2d(in_channels=features, out_channels=C_out, kernel_size=1, stride=1, padding=0, bias=False)
        
    
    def forward(self, x):
        x=self.backbone(x)
        x=self.resnet(x)
        x=self.end(x)

        return x
    
    

