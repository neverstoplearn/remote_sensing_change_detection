#!/usr/bin/env python
# coding: utf-8

# In[28]:
from PIL import Image
import cv2
from path import Path
import collections
from ranger import Ranger
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from utils.datasets import ATDataset
from utils.loss import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from torchvision.transforms import Resize, CenterCrop, Normalize
from utils.metrics import Metrics
#from utils.lr_scheduler import LR_Scheduler
from models.atcdnet.atcdnet import ATCDNet
#from models.appanet.APPANet import APPANet
from models.cdnet.cdnet import CDNet
from models.fcn.Fcn_cd import FCN_CD
from models.siamunet_conv.SiamUNet_Conv import SiamUnet_conc
from models.siamunet_diff.SiamUNet_Diff import SiamUnet_diff
from models.unet.UNet_CD import Unet
from models.nestunet.nestunet import NestedUNet_CD
from models.unet_aspp.UNet_ASPP import UNet_ASPP
from models.msaanet.MSAANet import Get_MSAANet
import datetime
import random
import os
import tqdm
import json
import argparse
from logsetting import  get_log
from torch_poly_lr_decay import PolynomialLRDecay
device = 'cuda'
path = './dataset'


def get_dataset_loaders(workers, batch_size = 4):
    target_size = 256

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    trainval_transform = transforms.Compose(
        [
#             JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            #JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            #JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(90),
            #transforms.RandomRotation(180),
            #transforms.RandomRotation(270),
            transforms.ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    target_transform = transforms.Compose(
        [
            #JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            # JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            # JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(90),
            #transforms.RandomRotation(180),
            #transforms.RandomRotation(270),
            transforms.ToTensor(),
            #Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            Normalize(mean=mean,std=std)
        ]
    )

    train_dataset = ATDataset(
        os.path.join(path, "train", "A"),os.path.join(path,"train","B"), os.path.join(path, "train", "OUT1"), trainval_transform,test_transform,target_transform
    )

    val_dataset = ATDataset(
        os.path.join(path, "val", "A"),os.path.join(path,"val","B"), os.path.join(path, "val", "OUT1"), trainval_transform,test_transform,target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)

    return train_loader, val_loader


def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.train()
    for images1,images2, masks  in tqdm.tqdm(loader):
        images1 = images1.to(device)
        images2 = images2.to(device)
        masks = masks.to(device)

        assert images1.size()[2:] == images2.size()[2:] == masks.size()[2:], "resolutions for images and masks are in sync"

        num_samples += int(images1.size(0))
        #print(num_samples)
        optimizer.zero_grad()
        outputs = net(images1,images2)
        #print(outputs.shape,masks.shape)
        #masks = masks.view(batch_size,masks.size()[2],masks.size()[3])
        #print(masks.shape)
        #masks = masks.squeeze()
        
        assert outputs.size()[2:] == masks.size()[2:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks.float()) ##BCELoss
        #loss = criterion(outputs, masks.long())
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    assert num_samples > 0, "dataset contains training images and labels"

    return {
        "loss": running_loss / num_samples,
        "precision": metrics.get_precision(),
        "recall": metrics.get_recall(),
        "f_score": metrics.get_f_score(),
        "oa":metrics.get_oa()
    }

def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.eval()

    for images1, images2, masks,  in tqdm.tqdm(loader):
        images1 = images1.to(device)
        images2 = images2.to(device)
        masks = masks.to(device)

        assert images1.size()[2:] == images2.size()[2:] == masks.size()[2:], "resolutions for images and masks are in sync"

        num_samples += int(images1.size(0))

        outputs = net(images1,images2)

        assert outputs.size()[2:] == masks.size()[2:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks.float())  ##BCELoss
        #loss = criterion(outputs, masks.long())
        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output)

    assert num_samples > 0, "dataset contains validation images and labels"

    return {
        "loss": running_loss / num_samples,
        "precision": metrics.get_precision(),
        "recall": metrics.get_recall(),
        "f_score": metrics.get_f_score(),
        "oa":metrics.get_oa()
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', nargs='?', type=int, default=50,
                    help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                    help='Batch Size')
    parser.add_argument('--swa_start', nargs='?', type=int, default=1)
    parser.add_argument('--lr', nargs='?', type=float, default=5e-3, 
                    help='Learning Rate')
    parser.add_argument('--model',nargs='?',type=str,default='ATCDNet')
    parser.add_argument('--swa',nargs='?',type=bool,default=True)
    
    parser.add_argument('--start_epoch',default=1, type=int)
    parser.add_argument('--pretrain',default=True,type=bool)
    ###resume model
    parser.add_argument('--r',dest='resume',default=False,type=bool)
    

    arg = parser.parse_args()


    num_classes = 2
    model_name = arg.model
    print(model_name)
    learning_rate = arg.lr
    num_epochs = arg.n_epoch
    batch_size = arg.batch_size


    history = collections.defaultdict(list)
    model_dict = {
                #'APPANet':APPANet(nInputChannels=6, n_classes=2, os=8, pretrained=False, _print=True).train().to(device),
                'ATCDNet':ATCDNet(nInputChannels=6, n_classes=2, os=8, pretrained=False, _print=True).train().to(device),
                'UNet_ASPP':UNet_ASPP(n_channels=6, n_classes=2).train().to(device),
                'NestedUNet_CD':NestedUNet_CD(in_ch=6, out_ch=2).train().to(device),
                'Unet':Unet(input_nbr=6, label_nbr=2).train().to(device),
                'SiamUnet_conc':SiamUnet_conc(input_nbr=3, label_nbr=2).train().to(device),
                'SiamUnet_diff':SiamUnet_diff(input_nbr=3, label_nbr=2).train().to(device),
                #'CDNet':CDNet(in_ch = 6,out_ch =2).train().to(device),
                'FCN_CD':FCN_CD(in_ch = 6,out_ch =2).train().to(device),
                'MSAANet':Get_MSAANet(in_channel=6,out_channel=2).train().to(device)
                }

    net = model_dict[model_name]
    print(net)
    if torch.cuda.device_count() > 1:
        print("using multi gpu")
        net = torch.nn.DataParallel(net,device_ids = [0, 1, 2, 3])
    else:
        print('using one gpu')
    """
    if args.resume:
        load_name = os.path.join("./model/ATCDNet_2020-03-02_20.pth")
        print("loading checkpoint %s" %(laod_name))
        net = torch.load(load_name)
        arg.start_epoch = net['epoch']
    """
    #if arg.pretrain:
    #    print("The ckp has been loaded sucessfully ")
    #net = torch.load("./model/MSAANet_2020-03-31_87.pth") # load the pretrained model
    #criterion = FocalLoss2d().to(device)
    criterion = torch.nn.BCELoss().to(device)
    #criterion = torch.nn.CrossEntropyLoss().to(device)
    train_loader, val_loader = get_dataset_loaders(5, batch_size)
    #opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
    opt = Ranger(net.parameters(),lr=learning_rate)
    today=str(datetime.date.today())
    logger = get_log(model_name + today +'_log.txt')
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5,eta_min=4e-08)
    #scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
    #                                        args.n_epoch, len(train_loader), logger=logger,
    #                                        lr_step=args.lr_step)
    #
    scheduler = PolynomialLRDecay(opt, max_decay_steps=100, end_learning_rate=0.0001, power=2.0)



    for epoch in range(num_epochs):
        logger.info("Epoch: {}/{}".format(epoch + 1, num_epochs))
        scheduler.step()
        #scheduler(opt,i,.step()
        train_hist = train(train_loader, num_classes, device, net, opt, criterion)
        logger.info( ('loss={}'.format(train_hist["loss"]),
                     'precision={}'.format(train_hist["precision"]),
                     'recall={}'.format(train_hist["recall"]),
                     'f_score={}'.format(train_hist["f_score"]),
                      'oa={}'.format(train_hist["oa"])))

 
        for k, v in train_hist.items():
            history["train " + k].append(v)

        val_hist = validate(val_loader, num_classes, device, net, criterion)
        logger.info(('loss={}'.format(val_hist["loss"]),
                     'precision={}'.format(val_hist["precision"]),
                     'recall={}'.format(val_hist["recall"]),
                     'f_score={}'.format(val_hist["f_score"]),
                      'oa={}'.format(val_hist["oa"])))

        for k, v in val_hist.items():
            history["val " + k].append(v)


        checkpoint = 'model/{}_{}_{}.pth'.format(model_name,today,epoch)
        torch.save(net,checkpoint)
    json = json.dumps(history)
    f = open("model/{}_{}.json".format(model_name,today),"w")
    f.write(json)
    f.close()
