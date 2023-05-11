from bz2 import compress
from glob import glob
from inspect import stack
import json
from tifffile import TiffFile
from concurrent.futures import ThreadPoolExecutor, process
from re import A, L
from matplotlib.font_manager import weight_dict
from sympy import root
import tifffile as tiff
import cv2
import numpy as np
import cupy as cp
import os
import time 
from tqdm import tqdm
from multiprocessing import Process, pool
import random
import math
import struct
import cupy
import numba 
from numba import jit
from multiprocessing import Pool
import threading
import multiprocessing




class z_slice_c(object):
    
    def __init__(self,planes_num_0,z_y_x_p,planes_num,result_x,result_y,result_folder
                 ,tiles_num,input_folder,pre,mid,end):
        self.planes_num_0=planes_num_0
        self.z_y_x_p=z_y_x_p
        self.planes_num=planes_num
        self.result_x=result_x
        self.result_y=result_y
        self.result_folder=result_folder
        self.tiles_num=tiles_num
        self.input_folder=input_folder
        self.pre=pre
        self.mid=mid
        self.end=end

    def get_weght(self,y,x,dir):
        map=np.ones((y,x),dtype=np.float16)
        if(dir==1):
            for i in range(y):
                for k in range(x):
                    map[i][k]=max(i/y,k/x)
        if(dir==2):
            for i in range(y):
                for k in range(x):
                    map[i][k]=max(i/y,(x-k)/x)
                    
        if(dir==3):
            for i in range(y):
                for k in range(x):
                    map[i][k]=max((y-i)/y,k/x)
                    
        if(dir==4):
            for i in range(y):
                for k in range(x):
                    map[i][k]=max((y-i)/y,(x-k)/x)
        return map
            

    def weght_mat(self,tiles,cur_tile,y_length=64,x_length=64):
        
        radio_mat=np.ones((y_length,x_length),dtype=np.float16)
        posY=int(cur_tile[1:3])
        posX=int(cur_tile[5:7])
        pos1=tiles.get(cur_tile)
        up_tile="Z"+str(posY-1).zfill(2)+cur_tile[3:7]
        down_tile='Z'+str(posY+1).zfill(2)+cur_tile[3:7]
        left_tile=cur_tile[0:5]+str(posX+1).zfill(2)
        right_tile=cur_tile[0:5]+str(posX-1).zfill(2)
        LU_tile='Z'+str(posY-1).zfill(2)+"_Y"+str(posX+1).zfill(2)
        LD_tile='Z'+str(posY+1).zfill(2)+"_Y"+str(posX+1).zfill(2)
        RU_tile='Z'+str(posY-1).zfill(2)+"_Y"+str(posX-1).zfill(2)
        RD_tile='Z'+str(posY+1).zfill(2)+"_Y"+str(posX-1).zfill(2)
        
        if up_tile in tiles:
        

            pos2=tiles.get(up_tile)
            x_l=x_length-abs(pos1[0]-pos2[0])
            overlap_y=y_length-(pos1[1]-pos2[1])
            for i in range(overlap_y):
                if(pos1[0]<pos2[0]):
                    radio_mat[i][-x_l:]=radio_mat[i][-x_l:]*i/overlap_y
                else:
                    radio_mat[i][:x_l]=radio_mat[i][-x_l:]*i/overlap_y

        if down_tile in tiles:

            pos2=tiles.get(down_tile)
            x_l=x_length-abs(pos1[0]-pos2[0])
            overlap_y=y_length-abs(pos1[1]-pos2[1])
            for i in range(overlap_y):
                if(pos1[0]<pos2[0]):
                    radio_mat[y_length-i-1][-x_l:]=radio_mat[y_length-i-1][-x_l:]*i/overlap_y
                else:
                    radio_mat[y_length-i-1][:x_l]=radio_mat[y_length-i-1][-x_l:]*i/overlap_y
                        
                    
        if left_tile in tiles:

            pos2=tiles.get(left_tile)
            overlap_x=x_length-abs(pos1[0]-pos2[0])
            y_l=y_length-abs(pos1[1]-pos2[1])
            temp=np.zeros((overlap_x,y_l),dtype=np.float16)
            for i in range(overlap_x):
                temp[i]=i/overlap_x
            temp=temp.transpose(1,0)
            if(pos1[1]>pos2[1]):
                radio_mat[-y_l:,0:overlap_x]=radio_mat[-y_l:,0:overlap_x]*temp
            else:
                radio_mat[0:y_l,0:overlap_x]=radio_mat[0:y_l,0:overlap_x]*temp
                
        if right_tile in tiles:

            pos2=tiles.get(right_tile)
            overlap_x=x_length-abs(pos1[0]-pos2[0])
            y_l=y_length-abs(pos1[1]-pos2[1])
            temp=np.zeros((overlap_x,y_l),dtype=np.float16)
            for i in range(overlap_x):
                temp[overlap_x-i-1]=i/overlap_x
            temp=temp.transpose(1,0)
            if(pos1[1]>pos2[1]):
                radio_mat[-y_l:,x_length-overlap_x:]=radio_mat[-y_l:,x_length-overlap_x:]*temp
            else:
                radio_mat[-y_l:,x_length-overlap_x:]=radio_mat[-y_l:,x_length-overlap_x:]*temp
                
        
        if LU_tile in tiles:
            pos2=tiles.get(LU_tile)
            overlap_x=x_length-abs(pos1[0]-pos2[0])
            overlap_y=y_length-abs(pos1[1]-pos2[1])
            if(overlap_x>0 and overlap_y>0):
                radio_mat[0:overlap_y,0:overlap_x]=radio_mat[0:overlap_y,0:overlap_x]*self.get_weght(overlap_y,overlap_x,1)
                
        if RU_tile in tiles:
            pos2=tiles.get(RU_tile)
            overlap_x=x_length-abs(pos1[0]-pos2[0])
            overlap_y=y_length-abs(pos1[1]-pos2[1])
            if(overlap_x>0 and overlap_y>0):
                radio_mat[0:overlap_y,x_length-overlap_x:]=radio_mat[0:overlap_y,x_length-overlap_x:]*self.get_weght(overlap_y,overlap_x,2)
                
        if LD_tile in tiles:
            pos2=tiles.get(LD_tile)
            overlap_x=x_length-abs(pos1[0]-pos2[0])
            overlap_y=y_length-abs(pos1[1]-pos2[1])
            if(overlap_x>0 and overlap_y>0):
                radio_mat[y_length-overlap_y:,0:overlap_x]=radio_mat[y_length-overlap_y:,0:overlap_x]*self.get_weght(overlap_y,overlap_x,3)
                
        if RD_tile in tiles:
            pos2=tiles.get(RD_tile)
            overlap_x=x_length-abs(pos1[0]-pos2[0])
            overlap_y=y_length-abs(pos1[1]-pos2[1])
            if(overlap_x>0 and overlap_y>0):
                radio_mat[y_length-overlap_y:,x_length-overlap_x:]=radio_mat[y_length-overlap_y:,x_length-overlap_x:]*self.get_weght(overlap_y,overlap_x,4)
        
        
        
        
        return radio_mat

    def weght_mat_simple(self,y_length,x_length,y_r=0.2,x_r=0.06):
        radio_mat=np.ones((y_length,x_length),dtype=np.float16)

        y_l=int(y_length*y_r)
        x_l=int(x_length*x_r)
        for i in range(0,y_l):
            for k in range(x_length):
                if(k>x_l and k<x_length-x_l):
                    radio_mat[i,k]=i/y_l
                else:
                    radio_mat[i,k]=min(i/y_l,min((x_length-k)/x_l,k/x_l))
        for i in range(y_length-y_l,y_length):
            for k in range(x_length):
                if(k>x_l and k<x_length-x_l):
                    radio_mat[i,k]=(y_length-i)/y_l
                else:
                    radio_mat[i,k]=min((y_length-i)/y_l,min((x_length-k)/x_l,k/x_l))
        for i in range(y_l,y_length-y_l):
            for k in range(x_length):
                if(k<x_l or k>x_length-x_l):
                    radio_mat[i,k]=min((x_length-k)/x_l,k/x_l)
        return radio_mat

    def quanzhi_map(self,index,planes_num,result_x,result_y,result_folder,tiles_num,input_folder,pre,mid,end):
        # z_ = len(str(planes_num))
        # dz_ = z_ - len(str(index))
        # z_str = str(index)
        # for i in range(dz_):
        #     z_str = '0' + z_str
        z_str = str(index).zfill(5)
        result_plane = np.zeros((result_y, result_x), dtype=np.float16)
        result_path = result_folder + '/z' + mid + z_str + '.tif'
        # print('result_path1:', result_path)
        l_ = len(str(tiles_num))
        tile_path_list = os.listdir(input_folder)
        #print(tiles_num)
        for i in range(tiles_num):
            # print('tile_i: ',i)
            # print("index:"+str(i))
            temp_t = self.z_y_x_p[i]
            l_str = temp_t['Tile']
            if index< temp_t['z'] or index>temp_t['z'] + self.planes_num_0-1:
                continue


            #
            # dl_ = l_ - len(str(i))
            # l_str = str(i)
            # for j in range(dl_):
            #     l_str = '0' + l_str
            
            z_str_i = str(index-temp_t['z']).zfill(5)
            # print(tile_path_list[i])
            img_path = input_folder + '/' +l_str+'/'+'z_'+z_str_i + '.tiff'
                # continue
            # print(img_path)
            # if(l_str[-2:]=='14'):
            #     img=np.zeros(img.shape,dtype=np.uint16)
            posX = temp_t['x']
            posY = temp_t['y']
            x_length = temp_t['x_length']
            y_length = temp_t['y_length']

            radio_map=self.weght_mat_simple(y_length,x_length)

            # pre_result_img=result_plane[posY:posY+y_length,posX:posX+x_length]
            # compare_img = np.zeros((2,y_length, x_length), dtype=np.uint16)
            # compare_img[0]=pre_result_img
            # compare_img[1]=img
            # max_mip=np.max(compare_img,axis=0)
            # result_plane[posY:posY+y_length,posX:posX+x_length]=max_mip

            result_plane[posY:posY+y_length,posX:posX+x_length] += radio_map
            
                # if(x==4 and (np.any(result_plane[x][posY:posY+y_length,posX:posX+x_length]))):
                #     pre_result_img=result_plane[x][posY:posY+y_length,posX:posX+x_length]
                #     compare_img = np.zeros((2,y_length, x_length), dtype=np.uint16)
                #     compare_img[0]=pre_result_img
                #     compare_img[1]=img
                #     max_mip=np.max(compare_img,axis=0)
                #     result_plane[x][posY:posY+y_length,posX:posX+x_length]=max_mip


            # for j in range(x_length):
            #     for k in range(y_length):

            #         if result_plane[posY + k][posX + j] == 0:
            #             result_plane[posY + k][posX + j] = img[k][j]  # 先y后x
            #         else:
            #             jj = 0
            #             kk = 0
            #             if j < x_length / 2:
            #                 jj = j
            #             else:
            #                 jj = x_length - j
            #             if k < y_length / 2:
            #                 kk = k
            #             else:
            #                 kk = y_length - k
            #             radio = jj / x_length * (1/0.055)
            #             if radio > kk / y_length * (1/0.2):
            #                 radio = kk / y_length * (1/0.2)
            #             #print("radio:" + str(radio))
            #             # print('www',result_plane[posY + k][posX + j])
            #             result_plane[posY + k][posX + j] = result_plane[posY + k][posX + j] * (1 - radio) + img[k][j] * radio

        return result_plane

    def stitch_one_plane(self,index,planes_num,result_x,result_y,result_folder,tiles_num,input_folder,pre,mid,end,qz_map):
        # z_ = len(str(planes_num))
        # dz_ = z_ - len(str(index))
        # z_str = str(index)
        # for i in range(dz_):
        #     z_str = '0' + z_str
        z_str = str(index).zfill(5)
        print(z_str)
        result_path = result_folder + '/z' + mid + z_str + '.tiff'
        if(os.path.exists(result_path)):
            return 
        result_plane = np.zeros((result_y, result_x), dtype=np.float16)

        # print('result_path1:', result_path)
        l_ = len(str(tiles_num))
        tile_path_list = os.listdir(input_folder)
        #print(tiles_num)
        for i in range(tiles_num):
            # print('tile_i: ',i)
            # print("index:"+str(i))
            temp_t = self.z_y_x_p[i]
            l_str = temp_t['Tile']
            if index< temp_t['z'] or index>temp_t['z'] + self.planes_num_0-1:
                continue


            #
            # dl_ = l_ - len(str(i))
            # l_str = str(i)
            # for j in range(dl_):
            #     l_str = '0' + l_str
            # print(tile_path_list[i])
            img_path = input_folder + '/' +l_str+ '.tiff'
            tiff_img=TiffFile(img_path) 
                # continue
            # print(img_path)
            img = tiff_img.asarray(key=slice(index-temp_t['z'],index-temp_t['z']+1))
            img=img.astype(np.float16)
            # if(l_str[-2:]=='14'):
            #     img=np.zeros(img.shape,dtype=np.uint16)
            posX = temp_t['x']
            posY = temp_t['y']
            x_length = temp_t['x_length']
            y_length = temp_t['y_length']



            # pre_result_img=result_plane[posY:posY+y_length,posX:posX+x_length]
            # compare_img = np.zeros((2,y_length, x_length), dtype=np.uint16)
            # compare_img[0]=pre_result_img
            # compare_img[1]=img
            # max_mip=np.max(compare_img,axis=0)
            # result_plane[posY:posY+y_length,posX:posX+x_length]=max_mip
            radio_map=self.weght_mat_simple(y_length,x_length)
            # mudle=np.ones(img.shape)
            # mudle=mudle*random.randrange(180,250 , 20)
            result_plane[posY:posY+y_length,posX:posX+x_length] +=radio_map*img
            # result_plane[posY:posY+y_length,posX:posX+x_length] +=radio_map*mudle/qz_map
            
                # if(x==4 and (np.any(result_plane[x][posY:posY+y_length,posX:posX+x_length]))):
                #     pre_result_img=result_plane[x][posY:posY+y_length,posX:posX+x_length]
                #     compare_img = np.zeros((2,y_length, x_length), dtype=np.uint16)
                #     compare_img[0]=pre_result_img
                #     compare_img[1]=img
                #     max_mip=np.max(compare_img,axis=0)
                #     result_plane[x][posY:posY+y_length,posX:posX+x_length]=max_mip


            # for j in range(x_length):
            #     for k in range(y_length):

            #         if result_plane[posY + k][posX + j] == 0:
            #             result_plane[posY + k][posX + j] = img[k][j]  # 先y后x
            #         else:
            #             jj = 0
            #             kk = 0
            #             if j < x_length / 2:
            #                 jj = j
            #             else:
            #                 jj = x_length - j
            #             if k < y_length / 2:
            #                 kk = k
            #             else:
            #                 kk = y_length - k
            #             radio = jj / x_length * (1/0.055)
            #             if radio > kk / y_length * (1/0.2):
            #                 radio = kk / y_length * (1/0.2)
            #             #print("radio:" + str(radio))
            #             # print('www',result_plane[posY + k][posX + j])
            #             result_plane[posY + k][posX + j] = result_plane[posY + k][posX + j] * (1 - radio) + img[k][j] * radio

        result_plane=result_plane/qz_map
        result_plane = result_plane.astype(np.uint16)
        tiff.imsave(result_path, result_plane,compression=5)

        return 


    # # pool = ThreadPoolExecutor(thread_num)
    # for i in range(5000,5001):
    #     mask=mask_map()
    #     qz_map=quanzhi_map()
    #     stitch_one_plane(i,planes_num,result_x,result_y,result_folder,tiles_num,input_folder,pre,mid,end,mask,qz_map)
    #     pass
    

    def work(self):
        qz_map=self.quanzhi_map(int(self.planes_num/2),self.planes_num,self.result_x,self.result_y,self.result_folder,self.tiles_num,self.input_folder,self.pre,self.mid,self.end)
        process_num=multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=process_num)
        for i in tqdm(range(0,self.planes_num)):
            pool.apply_async(self.stitch_one_plane, args=(i,self.planes_num,self.result_x,self.result_y,self.result_folder,self.tiles_num,self.input_folder,self.pre,self.mid,self.end,qz_map,))
            # self.stitch_one_plane(i,self.planes_num,self.result_x,self.result_y,self.result_folder,self.tiles_num,self.input_folder,self.pre,self.mid,self.end,qz_map)
        pool.close()
        pool.join()   
        print("All workers finished")
        
        

