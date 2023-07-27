import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, num_layers, inception, *params):
        super(ResNet, self).__init__()

        blk_list=[]
        for i in range(0, num_layers):
            blk_list.append(inception(params))
        self.blk_list = nn.Sequential(*list([b for b in blk_list]))

        self.num_layers=num_layers

    def forward(self,x):
        for i in range(self.num_layers):
            x=self.blk_list[i](x)+x

        return x