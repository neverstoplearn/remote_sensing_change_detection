import numpy as np
from skimage import io
from path import Path
import cv2
import torch
num_class = 2
dataset_path = ['/home/yons/zhengxin/ATCDnet/dataset/test2/OUT']
for item in dataset_path:
    file_path = [i for i in Path(f'{item}/').files() if 'jpg' in i.name]
    for im in file_path:
    
#path = "F:\\ATCDnet\\train\\OUT1\\10000.jpg"
        mask = io.imread(im).astype('int')
        mask_label = torch.zeros(num_class,mask.shape[0],mask.shape[1]).long()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                #print(temp[i][j])      
                if mask[i][j] > 129:
                    mask[i][j] = 1
                else:
                    mask[i][j] = 0      
                if mask[i][j] == 1:
                    mask_label[1][i][j] = 1
                else:
                    mask_label[0][i][j] = 1
        np.save('/home/yons/zhengxin/ATCDnet/dataset/' + im.split('.')[0].split('/')[-3] +'/OUT1/' + im.split('.')[0].split('/')[-1] + '.npy',mask_label)

        
