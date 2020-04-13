import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block_nested(nn.Module):
    def __init__(self,in_ch,mid_ch,out_ch):
        super(conv_block_nested,self).__init__()
        
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias = True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias = True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        return x
    
class NestedUNet_CD(nn.Module):
    def __init__(self,in_ch=6,out_ch=2):
        super(NestedUNet_CD,self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        
        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        
        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0],filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        
        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        
        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        
        self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size = 1)
        self.sigmoid = nn.Sigmoid()
        self.final2 = nn.Conv2d(out_ch * 4, out_ch, kernel_size = 1)
        
    def forward(self, x1, x2):
        x = torch.cat((x1,x2),1)
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)],1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)],1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)],1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)],1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)],1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)],1))
    
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)],1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)],1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)],1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)],1))
        
        x0_1 = self.final1(x0_1)
        y0_1 = self.sigmoid(x0_1)
        x0_2 = self.final1(x0_2)
        y0_2 = self.sigmoid(x0_2)
        x0_3 = self.final1(x0_3)
        y0_3 = self.sigmoid(x0_3)
        x0_4 = self.final1(x0_4)
        y0_4 = self.sigmoid(x0_4)
        
        y0_5 = self.sigmoid(self.final2(torch.cat([y0_1, y0_2, y0_3, y0_4],1)))
        
        
        
        return y0_5
"""    
from torchsummary import summary
model = NestedUNet_CD(in_ch = 6,out_ch =2)
summary(model,input_size=[(3,256,256),(3,256,256)],batch_size = 2, device="cpu")
"""