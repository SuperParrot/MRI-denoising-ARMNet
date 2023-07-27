import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio
import random

from models.ADP_Branch import Net as ADP_Branch
from models.DCT_Branch import Net as DCT_Branch
from models.MultiDilaResnet import MultiDilaResnet

from models.losses import *

class Net(nn.Module):
    def __init__(self, noise_level=None):
        super(Net, self).__init__()
        
        if(torch.cuda.is_available()):
            self.device=torch.device("cuda")
        else:
            self.device=torch.device("cpu")

        self.adp_branch=ADP_Branch()
        self.dct_branch=DCT_Branch()

        self.resnet_mask = MultiDilaResnet(C_in=4, C_out=4, num_layers=9)
        self.resnet_ori = MultiDilaResnet(C_in=4, C_out=4, num_layers=6)
        self.resnet_adp = MultiDilaResnet(C_in=4, C_out=4, num_layers=6)
        self.resnet_lowFreq=MultiDilaResnet(C_in=4, C_out=4, num_layers=6)
        self.resnet_highFreq=MultiDilaResnet(C_in=4, C_out=4, num_layers=6)

        self.resnet_merge_adp=MultiDilaResnet(C_in=8, C_out=8, num_layers=5)
        self.resnet_merge_dct=MultiDilaResnet(C_in=12, C_out=8, num_layers=5)

        self.noise_level=noise_level
    
    def forward(self, x):
        B,C,H,W=x.shape

        #Apply Rician noise
        with torch.no_grad():
            x_noised, noise_sigma=self.apply_noise(x, self.noise_level)

        '''
        plt.imshow(x_noised[0,0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()
        exit(0)
        '''

        '''
        error=x_noised[0,0,15:49, 3:34]-x[0,0,15:49, 3:34]
        plt.hist(error.flatten().cpu(), bins=300)
        plt.show()
        error = x_noised[0, 0, 15:49, 3:34] - x[0, 0, 15:49, 3:34]
        print(noise_sigma, error.var())
        exit(0)
        '''

        noise_var = noise_sigma ** 2
        rayleigh_var=0.42920*noise_var
        rayleigh_bias=1.253*torch.sqrt(noise_var)

        x_noised_adpFil=self.adp_branch(x_noised, rayleigh_var, rayleigh_bias)
        mask = self.resnet_mask(x_noised_adpFil)
        mask = torch.sigmoid(mask)
        '''
        plt.imshow((x_noised * mask)[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()
        exit(0)
        '''

        '''
        plt.subplot(2, 2, 1)
        plt.imshow(x_noised[0, 0].cpu().detach().numpy())
        plt.subplot(2, 2, 2)
        plt.imshow(x_noised_adpFil[0, 0].cpu().detach().numpy())
        plt.subplot(2, 2, 3)
        plt.imshow(mask[0, 0].cpu().detach().numpy())
        plt.subplot(2, 2, 4)
        plt.imshow((x_noised * mask)[0, 0].cpu().detach().numpy())
        plt.show()
        exit(0)
        '''

        x_adpFil = self.adp_branch(x_noised*mask, noise_var, 0)
        x_dctLowFreq, x_dctHighFreq=self.dct_branch(x_noised*mask)
        '''
        plt.imshow(x_dctHighFreq[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()
        exit(0)
        '''

        '''
        x_tmp=x_dctLowFreq+x_dctHighFreq
        print(torch.abs(x_adpFil - x_tmp).mean())
        plt.subplot(1, 2, 1)
        plt.imshow(x_adpFil[0, 0].cpu().detach().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(x_tmp[0, 0].cpu().detach().numpy())
        plt.show()
        exit(0)
        '''

        '''
        print(torch.abs(x_noised - x).mean())
        print(torch.abs(x_adpFil - x).mean())
        plt.subplot(1, 2, 1)
        plt.imshow(x_adpFil[0, 0].cpu().detach().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(torch.abs(x_adpFil - x)[0, 0].cpu().detach().numpy())
        plt.show()
        exit(0)
        '''

        '''
        print(torch.abs(x_noised-x).mean())
        print(torch.abs(x_dctLowFreq-x).mean())
        plt.subplot(1, 2, 1)
        plt.imshow(x_dctLowFreq[0, 0].cpu().detach().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow((x_dctLowFreq-x)[0, 0].cpu().detach().numpy())
        plt.show()
        exit(0)
        '''

        resnet_noised = self.resnet_ori(x_noised)
        resnet_adp = self.resnet_adp(x_adpFil)
        resnet_lowFreq = self.resnet_lowFreq(x_dctLowFreq)
        resnet_highFreq = self.resnet_highFreq(x_dctHighFreq)

        result_adp = self.resnet_merge_adp(torch.cat((resnet_noised, resnet_adp), dim=1))
        residue_adp, attention_adp = torch.chunk(result_adp, 2, dim=1)
        result_adp = x_adpFil + residue_adp
        result_adp = F.leaky_relu(result_adp, inplace=True)
        result_adp = torch.clamp(result_adp, -1, 1)

        result_dct = self.resnet_merge_dct(torch.cat((resnet_noised, resnet_lowFreq, resnet_highFreq), dim=1))
        residue_dct, attention_dct = torch.chunk(result_dct, 2, dim=1)
        result_dct = x_dctLowFreq + residue_dct
        result_dct = F.leaky_relu(result_dct, inplace=True)
        result_dct = torch.clamp(result_dct, -1, 1)

        attention_adp=torch.sigmoid(attention_adp)
        attention_dct=torch.sigmoid(attention_dct)
        attention_sum=attention_adp+attention_dct
        attention_adp=attention_adp/attention_sum
        attention_dct=attention_dct/attention_sum

        result = attention_adp * result_adp + attention_dct * result_dct
        result = F.leaky_relu(result, inplace=True)
        result = torch.clamp(result, -1, 1)

        '''
        plt.imshow(result[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(result[2, 0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(result[4, 0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()
        exit(0)
        '''

        return result_adp, result_dct, result, mask

    def apply_noise(self, x, noise_level):
        if (noise_level == None):
            noise_level = 0.20 * random.random()
        else:
            noise_level = self.noise_level
            assert noise_level < 1.0, 'noise_level must be smaller than 1.'
        noise_sigma = noise_level * x.max()
        # print(noise_sigma**2)

        N1 = torch.empty_like(x)
        if (torch.cuda.is_available()):
            N1 = N1.cuda()
        torch.normal(0, noise_sigma, out=N1)
        N2 = torch.empty_like(x)
        if (torch.cuda.is_available()):
            N2 = N2.cuda()
        torch.normal(0, noise_sigma, out=N2)
        x_noised = torch.sqrt((x + N1) ** 2 + N2 ** 2)

        return x_noised, noise_sigma