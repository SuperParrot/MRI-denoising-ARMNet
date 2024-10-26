import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio
import random
from thop import profile
from thop import clever_format

from models.ADP_Branch import Net as ADP_Branch
from models.DCT_Branch import Net as DCT_Branch
from models.UNet import UNet
from models.MultiDilaResnet import MultiDilaResnet

from models.losses import *


class Net(nn.Module):
    def __init__(self, noise_level=None):
        super(Net, self).__init__()

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.adp_branch = ADP_Branch()
        self.dct_branch = DCT_Branch()

        self.mask_cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),

            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.resnet_mask = MultiDilaResnet(C_in=4 * 2, C_out=64, num_layers=5, height=64, width=64)

        self.resnet_adp = MultiDilaResnet(C_in=4 * 2, C_out=64, num_layers=5, height=64, width=64)

        self.resnet_lowFreq = MultiDilaResnet(C_in=4 * 2, C_out=64, num_layers=3, height=64, width=64)
        self.resnet_highFreq = MultiDilaResnet(C_in=4 * 2, C_out=64, num_layers=3, height=64, width=64)

        self.end_merge_att = MultiDilaResnet(C_in=64 * 4, C_out=4, num_layers=3, height=64, width=64)
        self.end_merge_residue = MultiDilaResnet(C_in=64 * 4, C_out=4, num_layers=3, height=64, width=64)

        '''
        macs_tot, params_tot = 0, 0
        x = torch.randn(1, 4, 64, 64)
        macs, params = profile(self.mask_cnn, inputs=(x,))
        macs_tot+=macs
        params_tot+=params

        x = torch.randn(1, 8, 64, 64)
        macs, params = profile(self.resnet_mask, inputs=(x,))
        macs_tot += macs
        params_tot += params

        macs, params = profile(self.resnet_adp, inputs=(x,))
        macs_tot += macs
        params_tot += params
        
        x = torch.randn(1, 1)
        macs, params = profile(self.dct_branch.learnableDCTNet.noise_level_net, inputs=(x,))
        macs_tot += macs
        params_tot += params
        
        x = torch.randn(1, 8, 64, 64)
        macs, params = profile(self.resnet_lowFreq, inputs=(x,))
        macs_tot += macs
        params_tot += params

        macs, params = profile(self.resnet_highFreq, inputs=(x,))
        macs_tot += macs
        params_tot += params

        x=torch.randn(1, 64*4, 64, 64)
        macs, params = profile(self.end_merge_att, inputs=(x,))
        macs_tot += macs
        params_tot += params

        macs, params = profile(self.end_merge_residue, inputs=(x,))
        macs_tot += macs
        params_tot += params

        macs, params = clever_format([macs_tot, params_tot], "%.3f")
        print(macs, params)
        exit(0)
        '''

        self.noise_level = noise_level


    def forward(self, x, train_mask):
        # Apply Rician noise
        with torch.no_grad():
            x_noised, noise_sigma = self.apply_noise(x, self.noise_level)

        '''
        print(noise_sigma)
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
        rayleigh_var = 0.42920 * noise_var
        rayleigh_bias = 1.253 * torch.sqrt(noise_var)

        x_noised_adpFil = self.adp_branch(x_noised, rayleigh_var, rayleigh_bias)

        if(train_mask):
            mask=self.mask_cnn(x_noised_adpFil)
        else:
            with torch.no_grad():
                mask=self.mask_cnn(x_noised_adpFil)
        #x_noised_masked=x_noised*(nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(mask)>0.1)
        x_noised_masked=x_noised*(mask>0.5)

        '''
        plt.imshow(((mask > 0.1).float() - (torch.abs(x)>0).float())[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()
        '''

        '''
        plt.subplot(1,3,1)
        plt.imshow((x_noised)[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(mask[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow((x_noised * (mask>0.5))[0, 0].cpu().detach().numpy(), cmap='gray')
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

        if(noise_sigma<0.10):
            x_adpFil = self.adp_branch(x_noised, noise_var, 0)
            x_dctLowFreq, x_dctHighFreq = self.dct_branch(x_noised, torch.tensor([noise_sigma]))
        else:
            x_adpFil = self.adp_branch(x_noised_masked, noise_var, 0)
            x_dctLowFreq, x_dctHighFreq = self.dct_branch(x_noised_masked, torch.tensor([noise_sigma]))

        '''
        plt.imshow(x_adpFil[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()
        exit(0)
        '''

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

        residue_mask=self.resnet_mask(torch.cat((x_noised, x_noised_masked), dim=1))
        residue_adp = self.resnet_adp(torch.cat((x_adpFil, x_noised), dim=1))
        residue_lowFreq = self.resnet_lowFreq(torch.cat((x_adpFil, x_dctLowFreq), dim=1))
        residue_highFreq = self.resnet_highFreq(torch.cat((x_adpFil, x_dctHighFreq), dim=1))

        residue = torch.cat((residue_mask, residue_adp, residue_lowFreq, residue_highFreq), dim=1)
        affine_att = self.end_merge_att(residue)
        affine_att = F.sigmoid(affine_att)

        affine_residue = self.end_merge_residue(residue)

        result = x_noised * affine_att + affine_residue
        '''
        plt.subplot(1, 3, 1)
        plt.imshow(x_noised[0, 0].cpu().detach().numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(affine_att[0, 0].cpu().detach().numpy())
        plt.subplot(1, 3, 3)
        plt.imshow(affine_residue[0, 0].cpu().detach().numpy())
        plt.show()
        '''

        return result, mask

    def apply_noise(self, x, noise_level):
        if (noise_level == None):
            noise_level = 0.20 * random.random()
        else:
            noise_level = self.noise_level
            assert noise_level < 1.0, 'noise_level must be smaller than 1.'
        noise_sigma = noise_level * x.max()
        # print(noise_sigma**2)

        N1 = torch.normal(torch.zeros_like(x), noise_sigma * torch.ones_like(x))
        N2 = torch.normal(torch.zeros_like(x), noise_sigma * torch.ones_like(x))

        x_noised = torch.sqrt((x + N1) ** 2 + N2 ** 2)

        return x_noised, noise_sigma
