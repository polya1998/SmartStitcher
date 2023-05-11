from locale import currency
from tkinter import W
import tifffile as tiff
import numpy as np
import os
import cv2
from tqdm import tqdm
from tkinter import W
import tifffile as tiff
import numpy as np
import os
from tqdm import tqdm
import time
import math
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor



def makeSlice(cur_tile_part,name,output_path,start_z,z_length,changesize):
    tile_name=name[:7]
    if((os.path.exists(output_path+'/'+tile_name)) and len(os.listdir(output_path+'/'+tile_name))==z_length):
        return 0
    img=cur_tile_part
    sp=img.shape
    if(not(os.path.exists(output_path+'/'+tile_name))):
             os.makedirs(output_path+'/'+tile_name)
    print(sp[0])
    for i in tqdm(range(sp[0])):
        plane_img=img[i]
        h=int(sp[1]/changesize)
        w=int(sp[2]/changesize)
        plane_img=cv2.resize(plane_img,(w,h))
        output_name=output_path+'/'+tile_name+'/'+'z_'+str(i+start_z).zfill(5)+'.tiff'
        if( os.path.exists(output_name)):
            continue
        tiff.imsave(output_name,plane_img)

def work(cur_tile_part,name,save_dirpath,start_z,z_length):
        makeSlice(cur_tile_part,name,save_dirpath,start_z,z_length,1)