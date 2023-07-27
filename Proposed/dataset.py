import numpy as np
import torch
import scipy.io as scio
from PIL import Image
from PIL import ImageChops
import matplotlib.pyplot as plt
import random
import os
import traceback

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, transform=None, target_transform=None, rand_mirror=True, rand_shift=True, rand_rotate=True):
        super(MyDataset,self).__init__()
        
        self.data_names=self.read_dataNames(file_name)
        print('%d data has been succesfully loaded.'%len(self.data_names))
        
        self.transform = transform
        self.target_transform = target_transform
        
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
            input, label, mask=self.get_data(index)
            #print(input.shape, label.shape)
        except Exception as e:
            input, label, mask=torch.zeros([4,64,64]), torch.zeros([4,64,64]), torch.zeros([4,64,64])
            if(len(self.data_names)==0):
                print('No data detected.')
            else:
                print('Error occurred when loading data:')
                print(self.data_names[index%len(self.data_names)])
                traceback.print_exc()

        if self.transform is not None:
            input=self.transform(input)
            label=self.transform(label)
            mask=self.transform(mask)
        
        return input, label, mask

    def get_data(self, index):
        data_name=self.data_names[index%len(self.data_names)]
        input=[]
        label=[]
        if(data_name.endswith('.mat')):
            input_mat=scio.loadmat(data_name)
            
            #label
            slice=self.normalize(np.array(input_mat['flair_slice']))
            label.append(slice)
            
            slice=self.normalize(np.array(input_mat['t1_slice']))
            label.append(slice)
            
            slice=self.normalize(np.array(input_mat['t1ce_slice']))
            label.append(slice)
            
            slice=self.normalize(np.array(input_mat['t2_slice']))
            label.append(slice)
            
            label=np.array(label)
            
            #input
            input=label
        
        if(self.rand_mirror):
            if(random.random()<0.5):
                input=self.mirror(input)
                label=self.mirror(label)

        if(self.rand_shift):
            xoff=random.randint(-8,8)
            yoff=random.randint(-8,8)
            input=self.shift(input, xoff, yoff, circleShift=False)
            label=self.shift(label, xoff, yoff, circleShift=False)
        
        if(self.rand_rotate):
            angle_idx=random.randint(0,3)
            input=self.rotate(input, angle_idx)
            label=self.rotate(label, angle_idx)
        
        input=torch.from_numpy(input).float()
        label=torch.from_numpy(label).float()
        mask=(input>0).float()
        
        return input, label, mask

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

class PreProcessing():
    def __init__(self):        
        pass
    
    def process(self, save_path, eval_ratio=0.0, test_ratio=0.0):
        assert eval_ratio+test_ratio<1.0, 'The sum of eval ratio and test ratio could not be greater than 1.0.'
        
        data_names=[]
        #data_names.extend(self.find_data('F:/Dataset/MICCAI_BraTS_2018_Data/train/extracted/'))
        #data_names.extend(self.find_data('/data1/JBW/MICCAI_BraTS_2018_Data/train/extracted/'))
        #data_names.extend(self.find_data('/HOME/scz0678/run/data/MICCAI_BraTS_2018_Data/train/extracted/'))
        data_names.extend(self.find_data('E:/MICCAI_BraTS_2018_Data/train/extracted/'))

        tot=len(data_names)
        print('%d data has been succesfully detected.'%tot)
        
        random.shuffle(data_names)
        
        eval_list=data_names[0:int(np.floor(tot*eval_ratio))]
        data_names=data_names[int(np.floor(tot*eval_ratio)):]
        
        test_list=data_names[0:int(np.floor(tot*test_ratio))]
        data_names=data_names[int(np.floor(tot*test_ratio)):]

        self.writeToTxt(save_path, 'train_list.txt', data_names)
        print('%d data are divided into the train set.'%len(data_names))
        
        self.writeToTxt(save_path, 'eval_list.txt', eval_list)
        print('%d data are divided into the eval set.'%len(eval_list))
        
        self.writeToTxt(save_path, 'test_list.txt', test_list)
        print('%d data are divided into the test set.'%len(test_list))
        
        
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

