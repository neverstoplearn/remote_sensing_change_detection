from PIL import Image
import cv2
from path import Path
import os
from torchvision import transforms
import torch
path = './dataset'

img0 = Image.open(os.path.join(path,'test2/A/02151.jpg'))
img1 = Image.open(os.path.join(path,'test2/B/02151.jpg'))
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img0 = trans(img0)
img0 = img0.unsqueeze(0)
img1 = trans(img1)
img1 = img1.unsqueeze(0)
img0,img1 = img0.cuda(), img1.cuda()

net = torch.load("/home/yons/zhengxin/ATCDnet/model/ATCDNet_2020-03-06_49.pth")###
output = net(img0,img1)
print(output.shape)
probs = torch.max(output,1)[1]

mask = probs.cpu().numpy().reshape(256,256)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):   
        if mask[i][j] == 1:
            mask[i][j] = 255
        else:
            mask[i][j] = 0    

mask = mask.astype(int)
cv2.imwrite(f'./result/result_vis.jpg',mask)







