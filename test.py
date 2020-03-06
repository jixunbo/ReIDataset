import os
import os.path as osp
import csv
from shutil import copyfile
import cv2
import time
import configparser
import numpy as np
import random
# 1329,559,294,408
gt = np.array([-226,386, -226+298, 386+326])
im=cv2.imread('/Users/jixunbo/Downloads/Train_Opposite_view/OV_underground_20_persons/camera2/2396.jpeg')
print(im.shape)
img = im[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]
print(img.shape)
cv2.imwrite('/Users/jixunbo/Desktop/jj.jpeg',img)