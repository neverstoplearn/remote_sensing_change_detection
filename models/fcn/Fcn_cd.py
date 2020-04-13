import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_CD(nn.Module):
    def __init__(self,in_ch=6, out_ch=2):
        super(FCN_CD,self).__init__()
        filters = [64,128,256,512,4096]
        self.conv1 = nn.Conv2d(in_ch, filters[0], kernel_size = 3, padding = 1, stride = 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(filters[0], filters[0], kernel_size = 3, padding = 1, stride = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(filters[0], filters[1], kernel_size = 3, padding = 1, stride = 1)
        self.conv4 = nn.Conv2d(filters[1], filters[1], kernel_size = 3, padding = 1, stride = 1)
        self.conv5 = nn.Conv2d(filters[1], filters[2], kernel_size = 3, padding = 1, stride = 1)
        self.conv6 = nn.Conv2d(filters[2], filters[2], kernel_size = 3, padding = 1, stride = 1)
        self.conv7 = nn.Conv2d(filters[2], filters[3], kernel_size = 3, padding = 1, stride = 1)
        self.conv8 = nn.Conv2d(filters[3], filters[3], kernel_size = 3, padding = 1, stride = 1)
        self.conv9 = nn.Conv2d(filters[3], filters[4], kernel_size = 7, padding = 3, stride = 1)
        self.conv10 = nn.Conv2d(filters[4], filters[4], kernel_size = 1, stride = 1)
        self.conv11 = nn.Conv2d(filters[4], out_ch, kernel_size = 1, stride = 1)
        self.deconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size = 3,stride = 2, padding = 1, output_padding = 1)
        self.conv12 = nn.Conv2d(filters[3], out_ch, kernel_size = 1, padding = 0, stride = 1)
        self.deconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size = 4, padding = 1, stride = 2 )
        self.conv13 = nn.Conv2d(filters[2], out_ch, kernel_size = 3, padding = 1, stride = 1)
        self.deconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size = 8, stride = 8)
        self.softmax = nn.Softmax(dim = 1)
        
    
    def forward(self, x1, x2):
        x = torch.cat((x1,x2),1)
        #print(x.shape)
        x = self.pool(self.relu(self.conv2(self.relu(self.conv1(x)))))
        #print(x.shape)
        x = self.pool(self.relu(self.conv4(self.relu(self.conv3(x)))))
        #print(x.shape)
        x1 = self.pool(self.relu(self.conv6(self.relu(self.conv5(x)))))
        #print(x1.shape)
        x2 = self.pool(self.relu(self.conv8(self.relu(self.conv7(x1)))))
        #print(x2.shape)
        x3 = self.pool(self.relu(self.conv8(self.relu(self.conv8(x2)))))
        #print(x3.shape)
        x4 = self.dropout(self.relu(self.conv9(x3)))
        #print(x4.shape)
        x5 = self.dropout(self.relu(self.conv10(x4)))
        #print(x5.shape)
        x6 = self.deconv1(self.conv11(x5))
        #print(x6.shape)
        x7 = self.conv12(x2)
        #print(x7.shape)
        x8 = self.deconv2(x7 + x6)
        print(x8.shape)
        x9 = self.conv13(x1)
        print(x9.shape)
        x10 = self.deconv3(x8 + x9)
        final = self.softmax(x10)
                       
        return final
                       
"""
from torchsummary import summary
model = FCN_CD(in_ch = 6,out_ch =2)
summary(model,input_size=[(3,256,256),(3,256,256)],batch_size = 2, device="cpu")  
"""