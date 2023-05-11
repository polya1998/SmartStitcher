import json
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import tifffile
import cv2 as cv
import numpy as np
import os
import sys
import math
from tqdm import tqdm
from PIL import Image
from pylab import *

from skimage import data
from skimage.restoration import rolling_ball

from matplotlib import pyplot as plt
 
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import match_template

def Logtrans(img,c):
    ir, ic = img.shape
    res = np.zeros((ir,ic),dtype=np.uint8)
    for imgr in range(ir):
        for imgc in range(ic):
            res[imgr,imgc] =np.uint8(c * np.log(1.0 + img[imgr,imgc]) + 0.5)
    return res
    
def equalHist(img, z_max = 255): # z_max = L-1 = 255
    # 灰度图像矩阵的高、宽
    H, W = img.shape
    # S is the total of pixels
    S = H * W

    out = np.zeros(img.shape)
    sum_h = 0
    for i in range(256):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime

    out = out.astype(np.uint8)
    return out

def image_pre_pocess(img1):
    img1=Logtrans(img1,20)

    # background =rolling_ball(img1,radius=200)

    # filtered_image = img1 - background

    # clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # # 限制对比度的自适应阈值直方图均衡化
    # img_gray_clahe = clahe.apply(img1)

    return img1

def shift_compute_up_down(image1,image2,x_shift_d_,y_shift_d,y_range):



    img1=image_pre_pocess(image1)
    img2=image_pre_pocess(image2)
    sp=img2.shape

    src=img1[sp[0]-y_range:sp[0],:]
    dst=img2[0:y_shift_d,x_shift_d_:sp[1]-x_shift_d_]

    result = match_template(src, dst)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    x_shift=x-x_shift_d_
    y_shift=y_range-y
    print(y_shift,x_shift,np.max(result))
    # 计算ncc
    # fig = plt.figure(figsize=(8, 3))
    # ax1 = plt.subplot(1, 3, 1)
    # ax2 = plt.subplot(1, 3, 2)
    # ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    # ax1.imshow(dst, cmap=plt.cm.gray)
    # ax1.set_axis_off()
    # ax1.set_title('template')

    # ax2.imshow(src, cmap=plt.cm.gray)
    # ax2.set_axis_off()
    # ax2.set_title('image')
    # # highlight matched region
    # hcoin, wcoin = dst.shape
    # rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    # ax2.add_patch(rect)

    # ax3.imshow(result)
    # ax3.set_axis_off()
    # ax3.set_title('`match_template`\nresult')
    # # highlight matched region
    # ax3.autoscale(False)
    # ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    # plt.show()
    return [y_shift,x_shift,np.max(result)]


def shift_compute_left_right(image1,image2,y_shift_d_,x_shift_d,x_range):



    img1=image_pre_pocess(image1)
    img2=image_pre_pocess(image2)
    sp=img2.shape
    src=img1[:,sp[1]-x_range:sp[1]]
    dst=img2[y_shift_d_:sp[0]-y_shift_d_,0:x_shift_d]

    result = match_template(src, dst)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    x_shift=x_range-x
    y_shift=y-y_shift_d_
    print(y_shift,x_shift,np.max(result))
    # fig = plt.figure(figsize=(8, 3))
    # ax1 = plt.subplot(1, 3, 1)
    # ax2 = plt.subplot(1, 3, 2)
    # ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    # ax1.imshow(dst, cmap=plt.cm.gray)
    # ax1.set_axis_off()
    # ax1.set_title('template')

    # ax2.imshow(src, cmap=plt.cm.gray)
    # ax2.set_axis_off()
    # ax2.set_title('image')
    # # highlight matched region
    # hcoin, wcoin = dst.shape
    # rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    # ax2.add_patch(rect)

    # ax3.imshow(result)
    # ax3.set_axis_off()
    # ax3.set_title('`match_template`\nresult')
    # # highlight matched region
    # ax3.autoscale(False)
    # ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    # plt.show()
    return [y_shift,x_shift,np.max(result)]

def xy_get_shift(p1, p2, path1, path2, flag,shift1,shift2,range):

    
    print(path1)
    print(path2)
    img1 = tifffile.imread(path1)
    img2 = tifffile.imread(path2)
    sp=img1.shape
    shift=[]
    if(not flag):
        shift=shift_compute_up_down(img1,img2,shift1,shift2,range)
        loc = (p1, p2, int(sp[0]-shift[0]),int(shift[1]),float(shift[2]))
    if(flag):
        shift=shift_compute_left_right(img1,img2,shift1,shift2,range)
        loc = (p1, p2, int(shift[0]),int(sp[1]-shift[1]),float(shift[2]))
    
    print('y_shift:'+str(shift[0])+' x_shift:'+str(shift[1])+'\n')

    return loc

def z_get_shift(p1, p2, path1, path2, flag,shift1,shift2,range,z_length):

    
    print(path1)
    print(path2)
    img1 = tifffile.imread(path1)[int(z_length/4):int(3*z_length/4),:]
    img2 = tifffile.imread(path2)[int(z_length/4):int(3*z_length/4),:]
    sp=img1.shape
    shift=[]
    if(not flag):
        shift=shift_compute_left_right(img1,img2,shift1,shift2,range)
        loc = (p1, p2, int(shift[0]),int(sp[1]-shift[1]),float(shift[2]))

    if(flag):
        shift=shift_compute_left_right(img1,img2,shift1,shift2,range)
        loc = (p1, p2, int(shift[0]),int(sp[1]-shift[1]),float(shift[2]))



    return loc