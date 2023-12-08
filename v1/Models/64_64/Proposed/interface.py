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
        if(torch.cuda.is_available()):
            self.device=torch.device("cuda")
        else:
            self.device=torch.device("cpu")
            
        self.model_path='./params_saved'
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
        net=Net()
        if(torch.cuda.is_available()):
            net=net.to(self.device)
        
        if(not os.path.exists(self.model_path)):
            os.mkdir(self.model_path)
        torch.save(net.state_dict(), self.model_path+'/params.pkl')
        print('New model saved.')
        

    def train(self, epoch=1, learning_rate=1e-5, batch_size = 1, save_freq=1, shuffle=True, noise_level=None):
        self.net=Net(noise_level=noise_level)
        if(torch.cuda.is_available()):
            try:
                self.net.load_state_dict(torch.load(self.model_path+'/params.pkl'))
            except:
                self.generateModel()
        else:
            try:
                self.net.load_state_dict(torch.load(self.model_path+'/params.pkl', map_location='cpu'))
            except:
                self.generateModel()

        self.net.to(self.device)
        self.net.train()

        self.charbLoss=CharbonnierLoss()
        self.diceLoss=DiceLoss()
        
        self.optimizer=optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch, eta_min=1e-6)
        
        #print(torch.cuda.device_count())
        if(torch.cuda.is_available()):
            device_ids = range(torch.cuda.device_count())
            if(len(device_ids)>1):
                self.net = torch.nn.DataParallel(self.net, device_ids=[0,1,2])
            else:
                torch.cuda.set_device(0)
        
        self.train_data=MyDataset(file_name='train_list.txt', transform=None)
        batchNum=self.train_data.__len__()//batch_size
        if(self.train_data.__len__() % batch_size!=0):
            batchNum+=1

        progressPerPrint=1/(epoch*batchNum)
        progress=0

        acc_last=0
        acc_decreaseCnt=0
        last_time=-1

        for i in range(epoch):
            train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=shuffle, collate_fn = None)
            epoch_aveLoss=0
            current_loss=0
            current_loss_batch_cnt=0

            for batch_idx, (input, label, mask) in enumerate(train_loader):
                loss_rec, loss_dice=self.train_batch(input, label, mask)
                loss=loss_rec+loss_dice
                current_loss+=loss
                current_loss_batch_cnt+=1

                if(last_time>0):
                    speed=progressPerPrint/(time.time()-last_time)
                    eta=(1.0-progress)/speed
                    m, s = divmod(eta, 60)
                    h, m = divmod(m, 60)

                    if ((batch_idx % 100 == 0 and batch_idx > 0) or batch_idx >= batchNum - 1):
                        print("epoch:%d/%d batch:%d/%d\tloss:%.4lf %.4lf ETA:%02dH:%02dM:%02dS"%(i+1, epoch, batch_idx+1, batchNum, loss_rec, loss_dice, h,m,s))
                else:
                    print("epoch:%d/%d batch:%d/%d\tloss:%.4lf %.4lf ETA:??H:??M:??S"%(i+1, epoch, batch_idx+1, batchNum, loss_rec, loss_dice))
                progress+=progressPerPrint
                last_time=time.time()

                if((batch_idx % 10 == 0 and batch_idx > 0) or batch_idx >= batchNum - 1):
                    loss=current_loss.cpu().detach().numpy()
                    epoch_aveLoss+=loss
                    print("loss: %.8lf"%(loss/current_loss_batch_cnt))
                    
                    current_loss_batch_cnt=0
                    current_loss = 0

            print("epoch %d has finished. Average loss is: %.6lf"%(i+1, epoch_aveLoss/batchNum))

            self.lr_scheduler.step()
            
            if(i%save_freq==0 or i==epoch-1):
                print('Saving...')
                try:
                    torch.save(self.net.module.state_dict(), self.model_path+'/params.pkl')
                    torch.save(self.net.module.state_dict(), self.model_path+'/params_'+str(i)+'.pkl')
                except:
                    torch.save(self.net.state_dict(), self.model_path+'/params.pkl')
                    torch.save(self.net.state_dict(), self.model_path+'/params_'+str(i)+'.pkl')
                print('Saved.')
                

    def train_batch(self, input, label, mask):
        input=Variable(input, requires_grad=False)
        label=Variable(label, requires_grad=False)
        mask=Variable(mask, requires_grad=False)

        if(torch.cuda.is_available()):
            input=input.to(self.device)
            label=label.to(self.device)
            mask = mask.to(self.device)

        pred_adp, pred_dct, pred, mask_pred=self.net(input)
        #print(pred.shape, label.shape)
        #print(mask_pred.shape, mask.shape)
        #exit(0)
        
        assert pred.shape==label.shape, 'The shapes of prediction and label must be equal.'

        loss_rec=0
        pred_adp_mask = ((torch.abs(pred_adp) > 1e-4) + (label > 1e-4)) > 0
        loss_rec=loss_rec+self.charbLoss(pred_adp[pred_adp_mask], label[pred_adp_mask])
        pred_dct_mask = ((torch.abs(pred_dct) > 1e-4) + (label > 1e-4)) > 0
        loss_rec = loss_rec + self.charbLoss(pred_dct[pred_dct_mask], label[pred_dct_mask])
        pred_final_mask = ((torch.abs(pred) > 1e-4) + (label > 1e-4)) > 0
        loss_rec = loss_rec + self.charbLoss(pred[pred_final_mask], label[pred_final_mask])
        loss_rec = loss_rec/3.0

        loss_dice=self.diceLoss(mask_pred, mask)

        loss=loss_rec+loss_dice

        #print(self.charbLoss(pred, label))
        #print(loss)
        #exit(0)
        self.optimizer.zero_grad()
        loss.backward()

        '''
        for name, parms in self.net.named_parameters():
            if(not parms.grad == None):
                print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad.max())
        exit(0)
        '''

        self.optimizer.step()
        #exit(0)
        
        return loss_rec, loss_dice

    def eval(self, eval_loader, assigned_modelNames=None):
        self.net.eval()

        F_mae = MAE(reduction='sum')
        F_psnr = PSNR(data_range=1.0, reduction='sum')
        F_ssim = SSIM(reduction='sum')
        geometricAverage = GeometricAverage()

        rec_tot = 0
        mae_val, psnr_val, ssim_val = 0.0, 0.0, 0.0

        for batch_idx, (input, label, mask) in enumerate(eval_loader):
            input = Variable(input, requires_grad=False)
            label = Variable(label, requires_grad=False)
            mask = Variable(mask, requires_grad=False)

            if (torch.cuda.is_available()):
                input = input.to(self.device)
                label = label.to(self.device)
                mask = mask.to(self.device)

            if (assigned_modelNames != None):
                pred_list = []

                for model_name in assigned_modelNames:
                    if (torch.cuda.is_available()):
                        self.net.load_state_dict(torch.load(self.model_path + '/' + model_name))
                        torch.cuda.set_device(0)
                        self.net = self.net.to(self.device)
                    else:
                        self.net.load_state_dict(torch.load(self.model_path + '/' + model_name, map_location='cpu'))

                    with torch.no_grad():
                        _, _, pred, pred_mask = self.net(input)

                    pred_list.append(pred.cpu().detach().numpy())

                pred_tensor = torch.Tensor(pred_list)
                pred = geometricAverage(pred_tensor)

            else:
                with torch.no_grad():
                    _, _, pred, pred_mask = self.net(input)

            '''
            error_map=pred-label
            plt.subplot(1,3,1)
            plt.imshow(pred[0,0].cpu(), vmin=0, vmax=1)
            plt.subplot(1,3,2)
            plt.imshow(label[0,0].cpu(), vmin=0, vmax=1)
            plt.subplot(1,3,3)
            plt.imshow(torch.abs(error_map[0,0]).cpu(), vmin=0, vmax=0.5)
            plt.show()
            #exit(0)

            plt.hist(error_map.flatten().cpu(), bins=300)
            plt.show()
            exit(0)
            '''
            rec_tot += label.shape[0]

            mae_val += F_mae(pred, label).cpu().detach().numpy()

            psnr_val += F_psnr(pred.cpu().detach().numpy(), label.cpu().detach().numpy())
            ssim_val += F_ssim(pred.cpu().detach().numpy(), label.cpu().detach().numpy())

            #print(rec_tot)
            #print(mae_val/label.shape[0])
            #exit(0)

        mae_val /= rec_tot
        psnr_val /= rec_tot
        ssim_val /= rec_tot

        return mae_val, psnr_val, ssim_val

    def predict(self, file_name, assigned_modelNames=None, noise_level=None):
        seed = 3
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)

        self.net=Net(noise_level=noise_level)

        test_data=MyDataset(file_name=file_name, transform=None, rand_mirror=False, rand_shift=False, rand_rotate=False)
        test_loader = DataLoader(dataset=test_data, batch_size=5, shuffle=False)
    
        if(assigned_modelNames==None):
            if(torch.cuda.is_available()):
                self.net.load_state_dict(torch.load(self.model_path+'/params.pkl'))
                torch.cuda.set_device(0)
                self.net=self.net.to(self.device)
            else:
                self.net.load_state_dict(torch.load(self.model_path+'/params.pkl', map_location='cpu'))
    
        return self.eval(test_loader, assigned_modelNames)
        