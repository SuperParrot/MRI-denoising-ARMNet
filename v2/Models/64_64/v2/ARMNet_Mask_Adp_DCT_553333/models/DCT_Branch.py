import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio


class DCT2d(nn.Module):
    def __init__(self, H, W):
        super(DCT2d, self).__init__()

        self.dct_C = torch.zeros([H, W])
        N = W
        self.dct_C[0, :] = 1 * torch.sqrt(1 / N * torch.ones(H))

        for i in range(1, H):
            for j in range(W):
                self.dct_C[i, j] = torch.cos(torch.tensor(3.1415926 * i * (2 * j + 1) / (2 * N))) * torch.sqrt(
                    torch.tensor(2 / N))

        self.dct_C_t = torch.transpose(self.dct_C, 0, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(start_dim=0, end_dim=1)

        y = torch.zeros_like(x)

        if (torch.cuda.is_available()):
            self.dct_C = self.dct_C.cuda()
            self.dct_C_t = self.dct_C_t.cuda()
            y = y.cuda()

        for i in range(B * C):
            y_tmp = torch.mm(self.dct_C, x[i])
            y[i] = torch.mm(y_tmp, self.dct_C_t)

        y = y.reshape([B, C, H, W])
        return y


class IDCT2d(nn.Module):
    def __init__(self, H, W):
        super(IDCT2d, self).__init__()

        self.dct_C = torch.zeros([H, W])
        N = W
        self.dct_C[0, :] = 1 * torch.sqrt(1 / N * torch.ones(H))

        for i in range(1, H):
            for j in range(W):
                self.dct_C[i, j] = torch.cos(torch.tensor(3.1415926 * i * (2 * j + 1) / (2 * N))) * torch.sqrt(
                    torch.tensor(2 / N))

        self.dct_C_t = torch.transpose(self.dct_C, 0, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(start_dim=0, end_dim=1)

        y = torch.zeros_like(x)

        if (torch.cuda.is_available()):
            self.dct_C = self.dct_C.cuda()
            self.dct_C_t = self.dct_C_t.cuda()
            y = y.cuda()

        for i in range(B * C):
            y_tmp = torch.mm(self.dct_C_t, x[i])
            y[i] = torch.mm(y_tmp, self.dct_C)

        y = y.reshape([B, C, H, W])
        return y


class LearnableDCTNet(nn.Module):
    def __init__(self):
        super(LearnableDCTNet, self).__init__()

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load quantization matrix
        qmat_mat = scio.loadmat('./utils/matlab/Qmat/qmats.mat')
        self.qmat = nn.Parameter(torch.tensor(qmat_mat['qmat_64']).float(), requires_grad=True)

        self.qmat_unfold = nn.Unfold(kernel_size=(self.qmat.shape[1], self.qmat.shape[2]),
                                     stride=(self.qmat.shape[1], self.qmat.shape[2]))
        self.qmat_fold = nn.Fold(kernel_size=(self.qmat.shape[1], self.qmat.shape[2]),
                                 stride=(self.qmat.shape[1], self.qmat.shape[2]), output_size=(64, 64))

        self.qmat_dct2d = DCT2d(self.qmat.shape[1], self.qmat.shape[2])
        self.qmat_idct2d = IDCT2d(self.qmat.shape[1], self.qmat.shape[2])

        self.qmat64_weights = nn.Parameter(torch.ones(self.qmat.shape[0]), requires_grad=True)

        self.noise_level_net=nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),

            nn.Linear(16, 16),
            nn.GELU(),

            nn.Linear(16, 2+self.qmat64_weights.shape[0]),
        )

    def forward(self, x, noise_level):
        B, C, H, W = x.shape

        if(torch.cuda.is_available()):
            noise_level=noise_level.cuda()

        if (torch.cuda.is_available()):
            self.qmat = self.qmat.cuda()

            self.qmat64_weights = self.qmat64_weights.cuda()

        x_patched_qmat64 = self.qmat_unfold(x)
        x_patched_qmat64 = x_patched_qmat64.reshape(B * C, self.qmat.shape[1], self.qmat.shape[2],
                                                    (H // self.qmat.shape[1]) * (W // self.qmat.shape[2]))
        x_patched_qmat64 = x_patched_qmat64.permute(0, 3, 1, 2)

        y_qmat64_lowFreq, y_qmat64_highFreq = self.perform_dct(x_patched_qmat64, self.qmat_dct2d, self.qmat_idct2d,
                                                               self.qmat, self.qmat64_weights, noise_level)
        y_qmat64_lowFreq = y_qmat64_lowFreq.permute(0, 2, 3, 1)
        y_qmat64_lowFreq = y_qmat64_lowFreq.reshape(B, C * self.qmat.shape[1] * self.qmat.shape[2],
                                                    (H // self.qmat.shape[1]) * (W // self.qmat.shape[2]))
        y_qmat64_lowFreq = self.qmat_fold(y_qmat64_lowFreq)

        y_qmat64_highFreq = y_qmat64_highFreq.permute(0, 2, 3, 1)
        y_qmat64_highFreq = y_qmat64_highFreq.reshape(B, C * self.qmat.shape[1] * self.qmat.shape[2],
                                                      (H // self.qmat.shape[1]) * (W // self.qmat.shape[2]))
        y_qmat64_highFreq = self.qmat_fold(y_qmat64_highFreq)

        return y_qmat64_lowFreq, y_qmat64_highFreq

    def perform_dct(self, x, dct2d, idct2d, qmat, qmat_weights, noise_level):
        B, C, H, W = x.shape

        block_size_x = qmat.shape[1]
        block_size_y = qmat.shape[2]

        noise_level_refine=self.noise_level_net(noise_level)

        learned_qmat = torch.zeros_like(qmat[0])
        qmat_weights = F.softmax(qmat_weights, dim=0) * noise_level_refine[0] + noise_level_refine[1]
        for i in range(qmat.shape[0]):
            learned_qmat = learned_qmat + qmat_weights[i] * noise_level_refine[2+i] * qmat[i]
        '''
        plt.imshow(learned_qmat.cpu().detach().numpy())
        plt.show()
        print(qmat_weights)
        exit(0)
        '''
        learned_qmat = learned_qmat.expand([B, C, block_size_x, block_size_y])
        blocks_dct = dct2d(x)
        blocks_lowFreq = idct2d(blocks_dct * learned_qmat)
        blocks_highFreq = idct2d(blocks_dct * (1 - learned_qmat))

        return blocks_lowFreq, blocks_highFreq


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.learnableDCTNet = LearnableDCTNet()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
                if (m.bias != None):
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x, noise_level):
        x = self.learnableDCTNet(x, noise_level)

        return x



