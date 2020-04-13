import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class DCM(nn.Module):
    def __init__(self, in_C, out_C):
        super(DCM, self).__init__()
        self.ks = [1, 3, 5]
        if in_C == 2048:
            self.mid_C = in_C // 4
        if in_C == 1024:
            self.mid_C = in_C // 2
        if in_C == 512:
            self.mid_C = in_C
        if in_C == 256:
            self.mid_C = in_C
        if in_C == 128:
            self.mid_C = in_C
        if in_C == 64:
            self.mid_C = in_C
        self.ger_kernel_branches = nn.ModuleList()
        for k in self.ks:
            self.ger_kernel_branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(k),
                    nn.Conv2d(in_C, self.mid_C, kernel_size=1)
                )
            )

        self.trans_branches = nn.ModuleList()
        self.fuse_inside_branches = nn.ModuleList()
        for i in range(len(self.ks)):
            self.trans_branches.append(
                nn.Conv2d(in_C, self.mid_C, kernel_size=1)
            )
            self.fuse_inside_branches.append(
                nn.Conv2d(self.mid_C, self.mid_C, 1)
            )

        self.fuse_outside = nn.Conv2d(len(self.ks) * self.mid_C + in_C, out_C, 1)

    def forward(self, x, y):
        """
        x: 被卷积的特征
        y: 用来生成卷积核
        """
        feats_branches = [x]
        for i in range(len(self.ks)):
            kernel = self.ger_kernel_branches[i](y)
            kernel_single = kernel.split(1, dim=0)
            x_inside = self.trans_branches[i](x)
            x_inside_single = x_inside.split(1, dim=0)
            feat_single = []
            for kernel_single_item, x_inside_single_item \
                    in zip(kernel_single, x_inside_single):
                feat_inside_single = self.fuse_inside_branches[i](
                    F.conv2d(
                        x_inside_single_item,
                        weight=kernel_single_item.transpose(0, 1),
                        bias=None,
                        stride=1,
                        padding=self.ks[i] // 2,
                        dilation=1,
                        groups=self.mid_C
                    )
                )
                feat_single.append(feat_inside_single)
            feat_single = torch.cat(feat_single, dim=0)
            feats_branches.append(feat_single)
        return self.fuse_outside(torch.cat(feats_branches, dim=1))
        #return torch.cat((fuse_outside,x),dim=1)

class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x, x1):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        out = torch.cat((out,x1),dim=1)
        return out


class MSAANet(nn.Module):

    def __init__(self,in_channel,out_channel, block, num_block):
        super(MSAANet,self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size = 3, stride = 1, padding = 1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_1 = nn.Conv2d(3584, 2048, 1)
        self.dconv_up3 = double_conv(2048, 1024)
        self.conv_2 = nn.Conv2d(1536, 512, 1)
        self.dconv_up2 = double_conv(512, 256)
        self.conv_3 = nn.Conv2d(512, 256, 1)
        self.dconv_up1 = double_conv(256, 128)
        #self.dconv_up0 = double_conv(192, 128)

        self.dconv_last=nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64,out_channel,1)
        )
        self.cb1 = ContextBlock(inplanes=64, ratio=1. / 16., pooling_type='att')
        self.dcm1 = DCM(in_C = 64, out_C = 64)
        self.cb2 = ContextBlock(inplanes=256, ratio=1. / 16., pooling_type='att')
        self.dcm2 = DCM(in_C = 256, out_C = 256)
        self.cb3 = ContextBlock(inplanes=512, ratio=1. / 16., pooling_type='att')
        self.dcm3 = DCM(in_C=512, out_C=512)
        self.cb4 = ContextBlock(inplanes=512, ratio=1. / 16., pooling_type='att')
        self.dcm4 = DCM(in_C=1024, out_C=512)
        self.cb5 = ContextBlock(inplanes=512, ratio=1. / 16., pooling_type='att')
        self.dcm5 = DCM(in_C=2048, out_C=512)
        
        self.conv1_1x = nn.Conv2d(128,64,1)
        self.conv2_1x = nn.Conv2d(512,256,1)
        self.conv3_1x = nn.Conv2d(1024,512,1)
        self.conv4_1x = nn.Conv2d(1536,1024,1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x1,x2):
        x = torch.cat((x1,x2),dim=1)
        ###print(x.shape) [1 6 256 256]
        conv1 = self.conv1(x)
        ###print(conv1.shape) [1 64 256 256]
        temp=self.maxpool(conv1)
        temp_1 = temp
        ###print(temp.shape)  [1 64 128 128]  stage1
        temp = self.dcm1(temp,temp)
        temp = self.cb1(temp,temp_1)
        #print(temp.shape)
        temp = self.conv1_1x(temp)
        conv2 = self.conv2_x(temp)
        ###print(conv2.shape)##
        ###print(conv2.shape) [1 256 64 64]   stage2
        conv2_1 = conv2
        conv2 = self.dcm2(conv2,conv2)
        conv2 = self.cb2(conv2,conv2_1)
        ##print(conv2.shape)[1 512 64 64]
        conv2 = self.conv2_1x(conv2)
        conv3 = self.conv3_x(conv2)
        ###print(conv3.shape) ###[1 512 32 32]   stage3
        conv3_1 = conv3
        conv3 = self.dcm3(conv3,conv3)
        conv3 = self.cb3(conv3,conv3_1)
        conv3 = self.conv3_1x(conv3)
        conv4 = self.conv4_x(conv3)
        ##print(conv4.shape) ###[1 1024 16 16]  stage4
        conv4_1 = conv4
        conv4 = self.dcm4(conv4, conv4)
        ###print(conv4.shape)
        conv4 = self.cb4(conv4,conv4_1)
        ##print(conv4.shape)##[1 1536 16 16]
        conv4 = self.conv4_1x(conv4)
        bottle = self.conv5_x(conv4)
        ###print(bottle.shape) [1 2048 8 8]   stage5
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)
        botttle_1 = bottle
        bottle = self.dcm5(bottle, bottle)
        bottle = self.cb5(bottle,botttle_1)
        x = self.upsample(bottle)
        ###print(x.shape)  [1 2048 16 16]
        # print(x.shape)
        # print(conv4.shape)
        x = torch.cat([x, conv4], dim=1)
        ###print(x.shape)  [1 3072 16 16]
        x = self.conv_1(x)  ###[1 1024 16 16]
        x = self.dconv_up3(x) ###[1 512 16 16]
        ###print(x.shape)###
        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)
        ###print(x.shape)   [1 512 32 32]
        # print(x.shape)
        # print(conv3.shape) dim=1)###[1 1024 32 32]
        x = self.conv_2(x)###[1 512 32 32]
        x = self.dconv_up2(x) ###[1 256 32 32]
        x = self.upsample(x) ###[1 256 64 64]
        x = torch.cat([x, conv2], dim=1)###[1 512 64 64]
        x = self.conv_3(x)
        x = self.dconv_up1(x)
        x=self.upsample(x) ###[1 128 128 128]
        x=torch.cat([x,temp],dim=1) ###[1 192 128 128]
        out=self.dconv_last(x)
        #print(out.shape)
        #x = F.softmax(x,dim=1)
        out = torch.sigmoid(out)
        return out

def Get_MSAANet(in_channel=6,out_channel=2):
    return MSAANet(in_channel,out_channel, block=BottleNeck, num_block = [3,4,23,3])

if __name__ == "__main__":
     model = Get_MSAANet(6,2)
     model.eval()
     image1 = torch.randn(1, 3,256, 256)
     image2 = torch.randn(1, 3,256, 256)
     with torch.no_grad():
         output = model.forward(image1, image2)


