import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as scio
from PIL import Image
from PIL import ImageChops
import matplotlib.pyplot as plt
import random
import os
import traceback

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, rand_mirror=True, rand_shift=True, rand_rotate=True):
        super(MyDataset,self).__init__()
        
        self.data_names=self.read_dataNames(file_name)
        print('%d data has been succesfully loaded.'%len(self.data_names))

        
        self.rand_mirror=rand_mirror
        self.rand_shift=rand_shift
        self.rand_rotate=rand_rotate

    def read_dataNames(self, file_name):
        data_names=[]
        
        f=open('./dataDiv_list/'+file_name, 'r')
        for data_name in f.readlines():
            data_name=data_name.rstrip('\n').replace('\\','/')
            
            data_names.append(data_name)
        f.close()

        return data_names

    def normalize(self, img):
        img=img.astype(np.float32)
        img=img/img.max()
        return img

    def __getitem__(self, index):
        try:
            img_in, label, mask=self.get_data(index)
            #print(input.shape, label.shape)
        except Exception as e:
            img_in, label, mask=torch.zeros([4,64,64]), torch.zeros([4,64,64]), torch.zeros([4,64,64])
            if(len(self.data_names)==0):
                print('No data detected.')
            else:
                print('Error occurred when loading data:')
                print(self.data_names[index%len(self.data_names)])
                traceback.print_exc()
                exit(0)
        
        return img_in, label, mask

    def get_data(self, index):
        data_name=self.data_names[index%len(self.data_names)]

        input_mat=scio.loadmat(data_name)
        img_in=np.array(input_mat['im_slice'])
        del input_mat
        
        if(self.rand_mirror):
            if(random.random()<0.5):
                img_in=self.mirror(img_in)

        if(self.rand_shift):
            xoff=random.randint(-8,8)
            yoff=random.randint(-8,8)
            img_in=self.shift(img_in, xoff, yoff, circleShift=False)
        
        if(self.rand_rotate):
            angle_idx=random.randint(0,3)
            img_in=self.rotate(img_in, angle_idx)

        label = torch.from_numpy(img_in).float().unsqueeze(0)
        label = F.interpolate(label, size=(64, 64), mode='bicubic').squeeze(0)
        img_in = label

        mask=(torch.abs(img_in)>0).float()

        return img_in, label, mask

    def mirror(self, data):
        result=np.empty_like(data)
        for i in range(data.shape[0]):
            result[i,:,:]=np.fliplr(data[i,:,:])
        return result

    def shift(self, data, xoff, yoff, circleShift=True):
        result=np.empty_like(data)
        for i in range(data.shape[0]):
            Img=Image.fromarray(data[i,:,:])
            height, width = Img.size
            c = ImageChops.offset(Img,xoff,yoff)
            c=np.array(c)
            if(not circleShift):
                if(yoff<0):
                    c[height-abs(yoff):,:]=0
                else:
                    c[0:abs(yoff),:]=0
                
                if(xoff<0):
                    c[:,width-abs(xoff):]=0
                else:
                    c[:,0:abs(xoff)]=0
            
            result[i,:,:]=c
        
        return result

    def rotate(self, data, angle_idx):
        if(angle_idx==0):
            return data
        else:
            result=np.empty_like(data)
            for i in range(data.shape[0]):
                Img=Image.fromarray(data[i,:,:])
                if(angle_idx==1):
                    Img=Img.transpose(Image.ROTATE_90)
                elif(angle_idx==2):
                    Img=Img.transpose(Image.ROTATE_180)
                elif(angle_idx==3):
                    Img=Img.transpose(Image.ROTATE_270)
            
                result[i,:,:]=np.array(Img)
            
            return result

    def __len__(self):
        return len(self.data_names)

class GetDataList():
    def __init__(self):        
        pass
    
    def get_dataList(self, data_paths, save_path, save_name):
        data_names=[]
        #data_names.extend(self.find_data('F:/Dataset/MICCAI_BraTS_2018_Data/train/extracted/'))
        #data_names.extend(self.find_data('/data1/JBW/MICCAI_BraTS_2018_Data/train/extracted/'))
        #data_names.extend(self.find_data('/HOME/scz0678/run/data/MICCAI_BraTS_2018_Data/train/extracted/'))
        for data_path in data_paths:
            data_names.extend(self.find_data(data_path))

        tot=len(data_names)
        print('%d data has been succesfully detected.'%tot)

        self.writeToTxt(save_path, save_name, data_names)
        
        
    def writeToTxt(self, save_path, file_name, data_list):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path)
    
        if os.path.exists(save_path+'/'+file_name):
            print('Warning: %s has already existed.'%file_name)
            os.remove(save_path+'/'+file_name)
        f = open(save_path+'/'+file_name,'w')
        for data_name in data_list:
            f.write(data_name+'\n')
        f.close()

    def find_data(self, path):
        all_files=[]
        for f in os.listdir(path):
            if(not f.endswith('.mat')):
                continue
            
            all_files.append(path+'/'+f)
            
        return all_files

