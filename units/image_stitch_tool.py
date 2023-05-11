import os
import numpy as np
from PIL import Image
import tifffile as tiff
from skimage.restoration import rolling_ball
import cv2 as cv
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
    img1=Logtrans(img1,10)

    # background =rolling_ball(img1,radius=200)

    # filtered_image = img1 - background

    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值直方图均衡化
    img_gray_clahe = clahe.apply(img1)

    return img_gray_clahe







def stitch_images(folder_path, overlap_x, overlap_y,filelist):
    raw = len(filelist)
    col = len(filelist[0])
    first_image = tiff.imread(folder_path+'/'+filelist[0][0])
    height,width = first_image.shape
    h=height
    w=width
    overlap_x=int(overlap_x*width)
    overlap_y=int(overlap_y*height)
    width=(col-1)*(width-overlap_x)+width
    height=(raw-1)*(height-overlap_y)+height
    # 初始化大图
    stitched_image = np.zeros((height,width),dtype=np.uint16)
    for y in range (raw):
        for x in range(col):
            file_path=folder_path+'/'+filelist[y][x]
            paste_x = (w-overlap_x)*x
            paste_y= (h-overlap_y)*y
            image_t=tiff.imread(file_path)
            stitched_image[paste_y:paste_y+h,paste_x:paste_x+w]=image_t
            print(file_path)
    return stitched_image

    # for file in image_files[1:]:
    #     # 读入图片
    #     img = Image.open(file)
    #     # 计算拼接位置
    #     paste_x = x
    #     paste_y = y + overlap_y
    #     # 将图片放入大图
    #     stitched_image.paste(img, (paste_x, paste_y))
    #     # 更新下一张拼接图片的位置
    #     x = paste_x + img.size[0] - overlap_x
    #     y = paste_y
    # return np.array(stitched_image)




def stitch_list(folder_path,rangemethod,col,row):
    file_name_list=os.listdir(folder_path)
    file_name_list = np.sort(file_name_list)
    file_pos=0
    locations=[]
    if(rangemethod=='1-1'):
        for y in range(0,row,1):
            location = []
            for x in range(0,col,1):
                tile_name=file_name_list[y*col+x][:-5]
                location.append(tile_name)
            locations.append(location)
    if(rangemethod=='1-2'):
        for y in range(0,row,1):
            location = []
            for x in range(col-1,-1,-1):
                tile_name=file_name_list[y*col+x][:-5]
                location.append(tile_name)
            locations.append(location)
    if(rangemethod=='1-3'):
        for y in range(row-1,-1,-1):
            location = []
            for x in range(0,col,1):
                tile_name=file_name_list[y*col+x][:-5]
                location.append(tile_name)
            locations.append(location)
    if(rangemethod=='1-4'):
        for y in range(row-1,-1,-1):
            location = []
            for x in range(col-1,-1,-1):
                tile_name=file_name_list[y*col+x][:-5]
                location.append(tile_name)
            locations.append(location)
    if(rangemethod=='2-1'):
        for y in range(0,row,1):
            location = []
            for x in range(0,col,1):
                tile_name=file_name_list[x*row+y][:-5]
                location.append(tile_name)
            locations.append(location)
    if(rangemethod=='2-2'):
        for y in range(0,row,1):
            location = []
            for x in range(col-1,-1,-1):
                tile_name=file_name_list[x*row+y][:-5]
                location.append(tile_name)
            locations.append(location)
    if(rangemethod=='2-3'):
        for y in range(row-1,-1,-1):
            location = []
            for x in range(0,col,1):
                tile_name=file_name_list[x*row+y][:-5]
                location.append(tile_name)
            locations.append(location)
    if(rangemethod=='2-4'):
        for y in range(row-1,-1,-1):
            location = []
            for x in range(col-1,-1,-1):
                tile_name=file_name_list[x*row+y][:-5]
                location.append(tile_name)
            locations.append(location)
    if(rangemethod=='3-1'):
        for y in range(0,row,1):
            location = []
            if(y%2==0):
                for x in range(0,col,1):
                    tile_name=file_name_list[y*col+x][:-5]
                    location.append(tile_name)
                locations.append(location)
            else:
                for x in range(col-1,-1,-1):
                    tile_name=file_name_list[y*col+x][:-5]
                    location.append(tile_name)
                locations.append(location)
    if(rangemethod=='3-2'):
        for y in range(0,row,1):
            location = []
            if(y%2==1):
                for x in range(0,col,1):
                    tile_name=file_name_list[y*col+x][:-5]
                    location.append(tile_name)
                locations.append(location)
            else:
                for x in range(col-1,-1,-1):
                    tile_name=file_name_list[y*col+x][:-5]
                    location.append(tile_name)
                locations.append(location)
    if(rangemethod=='3-3'):
        for y in range(row-1,-1,-1):
            location = []
            if(y%2==0):
                for x in range(0,col,1):
                    tile_name=file_name_list[y*col+x][:-5]
                    location.append(tile_name)
                locations.append(location)
            else:
                for x in range(col-1,-1,-1):
                    tile_name=file_name_list[y*col+x][:-5]
                    location.append(tile_name)
                locations.append(location)
    if(rangemethod=='3-4'):
        for y in range(row-1,-1,-1):
            location = []
            if(y%2==1):
                for x in range(0,col,1):
                    tile_name=file_name_list[y*col+x][:-5]
                    location.append(tile_name)
                locations.append(location)
            else:
                for x in range(col-1,-1,-1):
                    tile_name=file_name_list[y*col+x][:-5]
                    location.append(tile_name)
                locations.append(location)
    return locations
# data=stitch_list('G:/gui3.20/3.18/MIP/Z_MIP','1-2',26,47)
# print(data)
                
        
        
        