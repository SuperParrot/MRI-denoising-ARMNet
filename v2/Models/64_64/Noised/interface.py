import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import random
from tqdm import tqdm
import numpy as np

from dataset import MyDataset

from models.Model import Net

from utils.Metrics import *
from utils.GeometricAverage import *
from utils.ModelMerge import *
from models.losses import CharbonnierLoss, DiceLoss


class Interface():
    def __init__(self):
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_path = './params_saved'
        '''
        seed = 3
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)
        '''

    def generateModel(self):
        print('Creating new model...')
        net = Net()
        if (torch.cuda.is_available()):
            net = net.to(self.device)

        if (not os.path.exists(self.model_path)):
            os.mkdir(self.model_path)
        torch.save(net.state_dict(), self.model_path + '/params.pkl')
        print('New model saved.')


    def eval(self, eval_loader):
        self.net.eval()

        F_mae = MAE(reduction='sum')
        F_psnr = PSNR(data_range=1.0, reduction='sum')
        F_ssim = SSIM(reduction='sum')
        F_lpips = LPIPS(reduction='sum')
        geometricAverage = GeometricAverage()

        rec_tot = 0
        mae_val, psnr_val, ssim_val, lpips_val = 0.0, 0.0, 0.0, 0.0

        for batch_idx, (input, label, _) in enumerate(eval_loader):
            if (torch.cuda.is_available()):
                input = input.to(self.device)
                label = label.to(self.device)

            pred = self.net(input)

            '''
            # self.draw_result(label[0, 0].cpu(), cmap='gray', vmin=0, vmax=1)
            for channel_idx in range(0, 4):
                self.draw_result(pred[0, channel_idx].cpu(),save_name='./Figure_' + str(channel_idx) + '_1.png', cmap='gray', vmin=0,vmax=1)
                self.draw_result(torch.abs(pred[0, channel_idx] - label[0, channel_idx]).cpu(),save_name='./Figure_' + str(channel_idx) + '_2.png', vmin=0, vmax=0.2)
            exit(0)
            '''

            rec_tot += label.shape[0]

            mae_val += F_mae(pred, label).cpu().detach().numpy()

            psnr_val += F_psnr(pred.cpu().detach().numpy(), label.cpu().detach().numpy())
            ssim_val += F_ssim(pred.cpu().detach().numpy(), label.cpu().detach().numpy())
            lpips_val += F_lpips(pred, label)

            # print(rec_tot)
            # print(mae_val/label.shape[0])
            # exit(0)

        mae_val /= rec_tot
        psnr_val /= rec_tot
        ssim_val /= rec_tot
        lpips_val /= rec_tot

        return mae_val, psnr_val, ssim_val, lpips_val

    def predict(self, file_name, noise_level=None):
        seed = 3
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)

        self.net = Net(noise_level=noise_level)
        if (torch.cuda.is_available()):
            try:
                self.net.load_state_dict(torch.load(self.model_path + '/params.pkl'))
            except:
                self.generateModel()
        else:
            try:
                self.net.load_state_dict(torch.load(self.model_path + '/params.pkl', map_location='cpu'))
            except:
                self.generateModel()

        test_data = MyDataset(file_name=file_name, rand_mirror=False, rand_shift=False, rand_rotate=False)
        test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=1, pin_memory=False)

        return self.eval(test_loader)

    def draw_result(self, img, save_name=None, cmap=None, vmin=0.0, vmax=1.0):
        zoom_x=round(img.shape[1]*0.2873)
        zoom_height=round(img.shape[1]*0.1978)
        zoom_y = round(img.shape[0] * 0.5393)
        zoom_width = round(img.shape[0] * 0.2629)

        subimg=img[zoom_x:zoom_x+zoom_height, zoom_y:zoom_y+zoom_width]

        subimg=nn.Upsample(mode='bicubic', scale_factor=1.5)(subimg.unsqueeze(0).unsqueeze(0))
        subimg=subimg.squeeze(0).squeeze(0)

        img[img.shape[0]-subimg.shape[0]:, img.shape[1]-subimg.shape[1]:]=subimg

        fig, ax = plt.subplots(1, 1)
        if (cmap != None):
            plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            plt.imshow(img, vmin=vmin, vmax=vmax)
        plt.axis('off')

        rect=plt.Rectangle(
            (zoom_y, zoom_x),  # 矩形左下角
            zoom_width,  # width
            zoom_height,  # height
            color='r',
            alpha=1.0,
            fill=None,
            linewidth=2
        )
        ax.add_patch(rect)

        rect = plt.Rectangle(
            (img.shape[1]-subimg.shape[1], img.shape[0]-subimg.shape[0]),
            subimg.shape[1]-1,  # width
            subimg.shape[0]-1,  # height
            color='r',
            alpha=1.0,
            fill=None,
            linewidth=2
        )
        ax.add_patch(rect)

        if(save_name!=None):
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0.0)

        plt.show()

