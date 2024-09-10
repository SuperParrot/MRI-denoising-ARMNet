import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio
import random

from models.losses import *


class Net(nn.Module):
    def __init__(self, noise_level=None):
        super(Net, self).__init__()

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tmp = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.noise_level = noise_level


    def forward(self, x):
        # Apply Rician noise
        with torch.no_grad():
            x_noised, noise_sigma = self.apply_noise(x, self.noise_level)

        return x_noised

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
