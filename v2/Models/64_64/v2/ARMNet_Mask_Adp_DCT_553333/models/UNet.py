import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio

from models.MultiDilaResnet import MultiDilaResnet


class UNet(nn.Module):
    def __init__(self, C_in, C_out, num_stages, features=64):
        super().__init__()

        H, W = 64, 64

        self.backbone=nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )

        enc_list = []
        for i in range(0, num_stages):
            enc_list.append(nn.Sequential(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=2, stride=2, padding=0, bias=False),
                nn.GELU(),
            ))

            H//=2
            W//=2

        self.enc_list = nn.Sequential(*list([b for b in enc_list]))
        del enc_list

        self.dec_start=nn.ConvTranspose2d(in_channels=features, out_channels=features, kernel_size=2, stride=2, padding=0, bias=False)

        dec_list=[]
        for i in range(0, num_stages-1):
            H *= 2
            W *= 2

            dec_list.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=features*2, out_channels=features, kernel_size=2, stride=2, padding=0, bias=False),
                nn.GELU(),
            ))

        self.dec_list = nn.Sequential(*list([b for b in dec_list]))
        del dec_list

        self.end = nn.Conv2d(in_channels=features, out_channels=C_out, kernel_size=1, stride=1, padding=0, bias=False)
        
    
    def forward(self, x):
        x=self.backbone(x)

        enc_result=[x]
        for i in range(len(self.enc_list)):
            enc_result.append(self.enc_list[i](enc_result[-1]))

        dec_result = self.dec_start(enc_result.pop())
        for i in range(len(self.dec_list)):
            dec_result=torch.cat((dec_result, enc_result.pop()), dim=1)
            dec_result=self.dec_list[i](dec_result)
        #print(dec_result.shape)
        #exit(0)

        x=self.end(x+dec_result)

        return x
    
    

