import numpy as np
import glob
import tqdm
import os
import cv2 as cv
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from skimage import io
from .en_de import Encode_Decode

def cal_val_kappa(val_pred_dir = "./val_predict/*.tif",targetPath = './val_set')


    en_de_tool = Encode_Decode()
    imglist = glob.glob(val_pred_dir)
    num = len(imglist)
    metric = []
    imgs = []
    targets = []
    for i in tqdm.tqdm(range(num)):
        name = os.path.split(imglist[i])[-1]
        targetPath = targetPath + name
        img = np.array(en_de_tool.encode_segmap(io.imread(imglist[i])))
        target = np.array(en_de_tool.encode_segmap(io.imread(targetPath)))
        imgs.append(img.tolist()), targets.append(target.tolist())
        # matrix = confusion_matrix(y_true=target, y_pred=img)
        # metric.append(matrix)
        # matrix2 +=matrix.astype(np.uint64)
    imgs, targets =np.array(imgs).flatten(), np.array(targets).flatten()
    # matrix = confusion_matrix(y_true=targets, y_pred=imgs)
    kappa = cohen_kappa_score(targets,imgs)
    # print("kappa:{},matrix:{}".format(kappa, matrix))
    return kappa
