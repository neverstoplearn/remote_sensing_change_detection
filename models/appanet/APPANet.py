import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i] * dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                                           model_dir='D:/Projects/Segmentation/pretrained')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(nInputChannels=3, os=8, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation

        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class APPAP(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, os):
        super(APPAP, self).__init__()
        self.chanel_in = in_dim
        self.os = os
        self.query_conv = nn.Conv2d(in_channels=640, out_channels=128, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=640, out_channels=128, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=512, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        if self.os == 8:
            dilations = [1, 2, 3, 6]
        self.aspp1 = ASPP_module(inplanes=512, planes=128, dilation=dilations[0])
        self.aspp2 = ASPP_module(inplanes=512, planes=128, dilation=dilations[1])
        self.aspp3 = ASPP_module(inplanes=512, planes=128, dilation=dilations[2])
        self.aspp4 = ASPP_module(inplanes=512, planes=128, dilation=dilations[3])
        self.aspp5 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(512, 128, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(128), nn.ReLU())
        self.conv = nn.Conv2d(in_channels=640,out_channels=512,kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()### [2 512 32 32] m=2 C =  512 h = 32 w = 32
        #proj_query1 = self.query_conv(x)
        p_q1 = self.aspp1(x)
        #print(p_q1.shape)           [2 128 32 32]
        p_q2 = self.aspp2(x)
        #print(p_q2.shape)           [2 128 32 32]
        p_q3 = self.aspp3(x)
        #print(p_q3.shape)           [2 128 32 32]
        p_q4 = self.aspp4(x)
        #print(p_q4.shape)           [2 128 32 32]
        p_q5 = self.aspp5(x)
        #print(p_q5.shape)           [2 128 1 1]
        p_q5 = F.interpolate(p_q5, size=p_q4.size()[2:], mode='bilinear', align_corners=True)
        #print(p_q5.shape)           [2 128 32 32]
        proj_query = torch.cat((p_q1, p_q2, p_q3, p_q4, p_q5), dim=1)
        #print(proj_query.shape)      [2 640 32 32]
        proj_query = self.query_conv(proj_query).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        #print(proj_query.shape)      [2 1024 128]
        #proj_key1 = self.key_conv(x)
        p_k1 = self.aspp1(x)
        p_k2 = self.aspp2(x)
        p_k3 = self.aspp3(x)
        p_k4 = self.aspp4(x)
        p_k5 = self.aspp5(x)
        p_k5 = F.interpolate(p_k5, size=p_k4.size()[2:], mode='bilinear', align_corners=True)
        proj_key = torch.cat((p_k1, p_k2, p_k3, p_k4, p_k5), dim=1)
        proj_key = self.key_conv(proj_key).view(m_batchsize, -1, width * height)
        #print(proj_key.shape) [2 128 1024]
        energy = torch.bmm(proj_query, proj_key)
        #print(energy.shape) [2 1024 1024]
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        #print(proj_value.shape) [2 512 1024]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        #print(out.shape)  [2 512 1024]
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class APPAC(nn.Module):

    def __init__(self, in_dim, os):
        super(APPAC, self).__init__()
        self.chanel_in = in_dim
        self.os = os
        if self.os == 8:
            dilations = [1, 2, 3, 6]

        self.aspp1 = ASPP_module(inplanes=512, planes=128, dilation=dilations[0])
        self.aspp2 = ASPP_module(inplanes=512, planes=128, dilation=dilations[1])
        self.aspp3 = ASPP_module(inplanes=512, planes=128, dilation=dilations[2])
        self.aspp4 = ASPP_module(inplanes=512, planes=128, dilation=dilations[3])
        self.aspp5 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(512, 128, 1, stride=1, bias=False),
                               nn.BatchNorm2d(128), nn.ReLU())
        self.conv1 = nn.Conv2d(in_channels=640, out_channels=512, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        #print(x.shape) [2 512 32 32]
        p_q1 = self.aspp1(x)
        p_q2 = self.aspp2(x)
        p_q3 = self.aspp3(x)
        p_q4 = self.aspp4(x)
        p_q5 = self.aspp5(x)
        p_q5 = F.interpolate(p_q5, size=p_q4.size()[2:], mode='bilinear', align_corners=True)
        proj_query = torch.cat((p_q1, p_q2, p_q3, p_q4, p_q5), dim=1)
        proj_query = self.conv1(proj_query).view(m_batchsize, C, -1)
        #print(proj_query.shape) [2 512 1024]

        p_k1 = self.aspp1(x)
        p_k2 = self.aspp2(x)
        p_k3 = self.aspp3(x)
        p_k4 = self.aspp4(x)
        p_k5 = self.aspp5(x)
        p_k5 = F.interpolate(p_k5, size=p_k4.size()[2:], mode='bilinear', align_corners=True)
        proj_key = torch.cat((p_k1, p_k2, p_k3, p_k4, p_k5), dim=1)
        proj_key = self.conv1(proj_key).view(m_batchsize, C, -1).permute(0, 2, 1)
        #print(proj_key.shape)  [2 1024 612]
        energy = torch.bmm(proj_query, proj_key)
        #print(energy.shape) [2 512 512]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        #print(out.shape) [2 512 32 32]
        return out


class APPAHead(nn.Module):
    def __init__(self, in_channels, norm_layer, os):
        super(APPAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.appap = APPAP(inter_channels, os)
        self.appac = APPAC(inter_channels, os)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        ###print(feat1.shape) [2 512 32 32]
        appap_feat = self.appap(feat1)
        appap_conv = self.conv51(appap_feat)

        feat2 = self.conv5c(x)
        appac_feat = self.appac(feat2)
        appac_conv = self.conv52(appac_feat)

        feat_sum = appap_conv + appac_conv

        return feat_sum


class APPANet(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=2, os=8, aux=False, pretrained=False, _print=True):
        if _print:
            print("Constructing APPANet model...")
            print("Backbone: Resnet-101")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(APPANet, self).__init__()
        self.head = APPAHead(2048, nn.BatchNorm2d, os)
        # Atrous Convolution
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)

        self.conv1 = nn.Conv2d(512, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        # adopt [1x1, 48] for channel reduction
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(64, 48, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, x_1, x_2):
        x = torch.cat((x_1, x_2), 1)
        ###print(x.shape) [2 6 256 256]
        x1, x2, x3, x4 = self.resnet_features(x)
        ###print(x1.shape,x2.shape,x3.shape,x4.shape)[2 256 64 64] [2 512 32 32] [2 1024 32 32] [2 2048 32 32]

        x_sum = self.head(x4)
        #print(x_sum.shape)   [2 512 32 32]
        x = self.conv1(x_sum)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)  [2 256 32 32]
        x = F.interpolate(x, size=(int(math.ceil(x_1.size()[-2] / 4)),
                                   int(math.ceil(x_1.size()[-1] / 4))), mode='bilinear', align_corners=True)
        #print(x.shape) #[2 256 64 64]
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        #print(x1.shape)
        x = torch.cat((x, x1), dim=1)

        x = self.last_conv(x)
        x = F.interpolate(x, size=x_1.size()[2:], mode='bilinear', align_corners=True)
        #x = F.softmax(x,dim=1)
        x = F.sigmoid(x)
        #print(x.shape)
        return x



if __name__ == "__main__":
     model = APPANet(nInputChannels=6, n_classes=2, os=8, pretrained=False, _print=True)
     model.eval()
     image1 = torch.randn(1, 3,256, 256)
     image2 = torch.randn(1, 3,256, 256)
     with torch.no_grad():
         output = model.forward(image1, image2)
"""
from torchsummary import summary

model = APPANet(nInputChannels=6, n_classes=2, os=8, pretrained=False, _print=True)
summary(model, input_size=[(3, 256, 256), (3, 256, 256)], batch_size=2, device="cpu")
"""