from path import Path
import torch
import matplotlib.image as mping
import glob
from PIL import Image
import torch.utils.data
from torchvision.transforms import Compose, Normalize
from skimage import io
from torchvision.transforms import transforms
import numpy as np

class ATDataset(torch.utils.data.Dataset):

    def __init__(self, inputs1,inputs2, target, trainval_transform=None,test_transform=None,target_transform=None,test = False):
        super().__init__()

        self.trainval_transform = trainval_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.test = test
        #self.inputs1 =  Path(inputs1).files()
        image_path1 = glob.glob(inputs1 + '/*.jpg')
        #print(image_path1)
        image_path1.sort()
        self.image_path1 = image_path1
        image_path2 = glob.glob(inputs2 + '/*.jpg')
        image_path2.sort()
        self.image_path2 = image_path2
        #self.inputs2 = Path(inputs2).files()
        if self.test == False:
            #self.target = Path(target).files()
            target = glob.glob(target + '/*.npy')
            target.sort()
            self.target = target

    def __len__(self):
#         return len(self.target)
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)
        

    def __getitem__(self, i):

        images1 = mping.imread(self.image_path1[i])
        images2 = mping.imread(self.image_path2[i])
        #images1 = io.imread(self.image_path1[i])
        #images1 = Image.open(self.image_path1[i])
        #images2 = Image.open(self.image_path2[i])
        #print(images1.shape)
        images1 = Image.fromarray(images1)
        #print(images1.shape)
        #images2 = io.imread(self.image_path2[i])
        images2 = Image.fromarray(images2)
        #print(self.image_path1[i],self.image_path2[i],self.target[i])
        if self.test == False:
            #mask  = Image.open(self.target[i])
            #mask = io.imread(self.target[i])[:,:,0:1]
            #mask = Image.fromarray(mask)
            #mask = mping.imread(self.target[i])
            mask = np.load(self.target[i]).astype('int')
            #mask = Image.fromarray(mask)
            #mask = mask[np.newaxis, :, :,:]  # add new dim in any dim
            mask = torch.LongTensor(mask)
            #print(mask.shape)
            if self.trainval_transform and self.target_transform is not None:
                images1 = self.trainval_transform(images1)
                images2 = self.trainval_transform(images2)
                #mask = self.target_transform(mask)
                mask = mask
            #print(mask.type)
            return images1,images2, mask
            #return torch.from_numpy(images1.astype(np.float32)), torch.from_numpy(images2.astype(np.float32)), torch.from_numpy(mask).long()
        else:
            return self.test_transform(images1),self.test_transform(images2)
            #torch.from_numpy(images1.astype(np.float32)), torch.from_numpy(images2.astype(np.float32))
"""
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
from torch.utils.data import DataLoader
train_loader = DataLoader(ATDataset('F:\\ATCDnet\\dataset\\train\\A1',
                                    'F:\\ATCDnet\\dataset\\train\\B1',
                                    'F:\\ATCDnet\\dataset\\train\\OUT1',
                                    trainval_transform=transforms.Compose(
                                        [
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.RandomVerticalFlip(),
                                            #transforms.RandomRotation(90),
                                            #transforms.RandomRotation(180),
                                            #transforms.RandomRotation(270),
                                            transforms.ToTensor(),
                                            Normalize(mean=mean, std=std),
                                        ]
                                    ), test_transform=None,
                                    target_transform =transforms.Compose(
                                        [
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.RandomVerticalFlip(),
                                            #transforms.RandomRotation(90),
                                            #transforms.RandomRotation(180),
                                            #transforms.RandomRotation(270),
                                            transforms.ToTensor(),
                                            #Normalize(mean=mean, std=std),
                                        ]
                                    )),batch_size=1, shuffle=True, num_workers=0)
train_iter = iter(train_loader)
x1, x2, target = next(train_iter)
print(x1.shape,x2.shape,target.shape)
"""