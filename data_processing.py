
# coding: utf-8

# In[1]:


from path import Path
from matplotlib import pyplot as plt
import numpy as np
import skimage.io as io
import os
from PIL import Image
import cv2
import random
import shutil


    
def crop_by_sequence(image_path,img_class_path,crop_size_w,crop_size_h,prefix,save_dir ,same_scale = False):
    """
    image_path : the image you want to crop
    img_class_path: the mask you want to crop
    crop_size_h: the size of height you want 
    crop_size_w: the size of weight you want 
    save_dir: the dir you want to save 
    prefix: the special word you want to add
    same_scale: big or small to same
    """
    raw_img = io.imread(image_path,)[:,:,1:]
    raw_img_class = io.imread(img_class_path,)
    
    
    if same_scale == True:
        crop_size_w = crop_size_w * 2 
        crop_size_h = crop_size_h * 2
    
    print(raw_img.shape,raw_img_class.shape)


    h,w,c = raw_img.shape[0],raw_img.shape[1],raw_img.shape[2]

    index = 0
    x2,y2 = 0,0
    x0,y0 = 0,0
    while(y2<h):
        while(x2<w):
            x1 = x0
            x2 = x1 + crop_size_w
            y1 = y0
            y2 = y1 +crop_size_h


            if(x2>w or y2>h):
                x2 = min(x2,w)
                y2 = min(y2,h)
                if((x2-x1)>10 and (y2-y1)>10):
                    backgroud = np.zeros((crop_size_h,crop_size_w,c),dtype=np.uint8)
                    backgroud[:y2-y1,:x2-x1,:] = raw_img[y1:y2,x1:x2,:]
                    patch = backgroud

                    backgroud_label = np.zeros((crop_size_h,crop_size_w),dtype=np.uint8)
                    backgroud_label[:y2-y1,:x2-x1] = raw_img_class[y1:y2,x1:x2]
                    patch_label = backgroud_label
                else:
                    break
            else:
                patch = raw_img[y1:y2,x1:x2,:]
                patch_label = raw_img_class[y1:y2,x1:x2]
            #stride_h = auto_stride(patch_label)
            stride_h = crop_size_h
            stride_w = crop_size_w
            #print "current stride: ",stride_h
            x0 = x1 + stride_w
            
            if same_scale == True:
                patch = cv2.resize(patch,(int(crop_size_w/2), int(crop_size_h/2)))
                patch_label = cv2.resize(patch_label,(int(crop_size_w/2), int(crop_size_h/2)))
                
            success = cv2.imwrite(save_dir + f'/images/{prefix}_sequence_{index}.png',patch)
            success_1 = cv2.imwrite(save_dir + f'/labels/{prefix}_sequence_{index}.png',patch_label)

            if success == 1 and success_1 ==1 :
                pass
            else:
                print('seq_save err')
            index = index + 1
        x0,x1,x2 = 0,0,0
        y0 = y1 + stride_h

        
        
def crop_by_random(num,image_path,img_class_path,crop_size_w,crop_size_h,prefix,save_dir, same_scale = False ):
    """
    image_path : the image you want to crop
    img_class_path: the mask you want to crop
    crop_size_h: the size of height you want 
    crop_size_w: the size of weight you want 
    save_dir: the dir you want to save 
    prefix: the special word you want to add
    same_scale: big or small to same
    """
    if same_scale == True:
        crop_size_w = crop_size_w * 2 
        crop_size_h = crop_size_h * 2

    raw_img = io.imread(image_path,)[:,:,1:]
    raw_img_class = io.imread(img_class_path)
    print(raw_img.shape, raw_img_class.shape)
    h,w,c = raw_img.shape[0],raw_img.shape[1],raw_img.shape[2]
    index = 0 
    range_h = h - crop_size_h - 1
    range_w = w - crop_size_w - 1
    
    list_x = np.random.randint(low = 0, high = range_h, size = num)
    list_y = np.random.randint(low = 0, high = range_w, size = num)
    combine = list(zip(list_x,list_y))
    for i in combine:
        
        patch = raw_img[i[0]:i[0] + crop_size_h, i[1]:i[1] + crop_size_w,:]
        patch_label = raw_img_class[i[0]:i[0] + crop_size_h, i[1]:i[1] + crop_size_w]
        
        if same_scale == True:
            patch = cv2.resize(patch,(int(crop_size_w/2), int(crop_size_h/2)))
            patch_label = cv2.resize(patch_label,(int(crop_size_w/2), int(crop_size_h/2)))

        success = cv2.imwrite(save_dir + f'/images/{prefix}_random_{index}.png',patch)
        success_1 = cv2.imwrite(f'{save_dir}/labels/{prefix}_random_{index}.png',patch_label)

        if success == 1 and success_1 ==1 :
                pass
        else:
            print('random save err', success, success_1)

        index = index + 1



def generate(ds_file:list, num = 1000,split = 5, crop_size_h = 512, crop_size_w = 512, save_dir = 'dataset',string = '',  same_scale = False, ):
    """
    num: the number of pictures split by random crop
    split: trainset : validationset
    crop_size_h: the size of height you want 
    crop_size_w: the size of weight you want 
    save_dir: the dir you want to save 
    string: the special word you want to add
    same_scale: big or small to same
    """
    print(crop_size_h, crop_size_w)

    os.mkdir(f'./{save_dir}/')
    os.mkdir(f'./{save_dir}/training')
    os.mkdir(f'./{save_dir}/training/images')
    os.mkdir(f'./{save_dir}/training/labels')
    os.mkdir(f'./{save_dir}/validation')
    os.mkdir(f'./{save_dir}/validation/images')
    os.mkdir(f'./{save_dir}/validation/labels')
    
    for f in ds_file:
        images = [i for i in Path(f'{f}/').files() if  len(str(i.name)) == 45]
        if 'train' in f:
            ge_save_dir = save_dir + '/training'
        else:
            ge_save_dir = save_dir +'/validation'
        for i in range(len(images)):
            image_path = images[i]
            img_class_path =  f'{f}/' + f'{images[i].stem[:-4]}'+ '_label_mask.png'
            prefix = f"picture_{i}"
            prefix = string + prefix
            print(image_path)
            print(img_class_path)
            crop_by_random(num,image_path,img_class_path,crop_size_w,crop_size_h,prefix,ge_save_dir, same_scale = same_scale )
            crop_by_sequence(image_path,img_class_path,crop_size_w,crop_size_h,prefix,ge_save_dir,  same_scale = same_scale)
    
    
    if split == True:
        ## split the train dataset and validation dataset
        img_sample = random.sample(Path(f'./{save_dir}/training/images/').files(),len(Path(f'./{save_dir}/training/images/').files())//split )
        train_img_dir = f'./{save_dir}/training/images/'
        train_label_dir = f'./{save_dir}/training/labels/'
        val_img_dir = f'./{save_dir}/validation/images/'
        val_label_dir = f'./{save_dir}/validation/labels/'
        for i in sample:
            shutil.move(train_img_dir + i.name,f'{val_img_dir}{i.name}')
            shutil.move(train_label_dir + i.name ,f'{val_label_dir}{i.name}')

generate(ds_file = ['train_set', 'val_set'])

