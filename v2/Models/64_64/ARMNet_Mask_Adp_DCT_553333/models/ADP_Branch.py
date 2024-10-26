import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio

class LearnableAdpFilter(nn.Module):
    def __init__(self, kernel_size=3):
        assert kernel_size%2!=0, 'Kernel size must be an odd number.'
    
        super(LearnableAdpFilter, self).__init__()
        
        if(torch.cuda.is_available()):
            self.device=torch.device("cuda")
        else:
            self.device=torch.device("cpu")

        self.kernel_size=kernel_size
        self.filter_width=(kernel_size-1)//2
        
        
    def forward(self, x, noise_var, noise_bias):
        B,C,H,W=x.shape

        #print(self.kernel_size)
        unfold_op=nn.Unfold(kernel_size=self.kernel_size, padding=self.filter_width)

        '''
        plt.imshow(x[0, 0].cpu())
        plt.show()
        '''

        # Rayleigh denoise
        x_unfolded=unfold_op(x)
        x_unfolded=x_unfolded.reshape(B*C,self.kernel_size*self.kernel_size, H*W)
        #print(x.shape)

        S_var = torch.var(x_unfolded, dim=1, keepdim=False)
        mid_val = x_unfolded.median(dim=1).values

        x=x.reshape(B*C,H*W)

        y = x - noise_var / (S_var + 1e-10) * (x - mid_val + noise_bias)
        y = F.relu(y)

        y = y.reshape(B, C, H, W)

        return y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        if(torch.cuda.is_available()):
            self.device=torch.device("cuda")
        else:
            self.device=torch.device("cpu")

        self.learnableAdpFilter=LearnableAdpFilter(kernel_size=5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
                if(m.bias!=None):
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight.data, 1)
                if(m.bias!=None):
                    nn.init.constant_(m.bias.data, 0)
        
    
    def forward(self, x, noise_var, noise_bias):
        x=self.learnableAdpFilter(x, noise_var, noise_bias)

        return x
    
    

