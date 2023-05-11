from bz2 import compress
from inspect import stack
import json
from concurrent.futures import ThreadPoolExecutor
from matplotlib.font_manager import weight_dict
from sympy import root
import tifffile as tiff
from tifffile import TiffFile
import cv2
import numpy as np
import os
import time 
from tqdm import tqdm
from multiprocessing import Process
import random
import math
import struct
from multiprocessing import Pool
import time
from scipy import ndimage
import multiprocessing

def downsample_array(input_array, rescale_factors):
    num_dims = len(input_array.shape)

    # Create an empty array with the new shape
    new_shape = tuple(int(s / f) for s, f in zip(input_array.shape, rescale_factors))
    output_array = np.empty(new_shape, dtype=input_array.dtype)

    # Define a helper function to apply area interpolation on a 2D array
    def resize_2d(array_2d, factor_x, factor_y):
        return cv2.resize(array_2d, (int(array_2d.shape[1] / factor_y), int(array_2d.shape[0] / factor_x)), interpolation=cv2.INTER_AREA)

    if num_dims == 2:
        return resize_2d(input_array, rescale_factors[0], rescale_factors[1])
    elif num_dims == 3:
        for z in range(new_shape[0]):
            output_array[z] = resize_2d(input_array[int(z * rescale_factors[0])], rescale_factors[1], rescale_factors[2])
        return output_array
    else:
        raise ValueError("The input array must be either 2D or 3D.")

class output_terafly_c(object):
    def __init__(self,data,posjson_path,input_folder,result_folder,x_length,y_length,planes_num_0,rescale_factor,real_factors,mode):
        self.jsoninfo=data
        self.planes_num_0=planes_num_0
        self.real_factors=real_factors
        self.result_folder=result_folder     
        self.mode=mode
           
        if(not(os.path.exists(self.result_folder))):
            os.makedirs(self.result_folder)
        self.input_folder=input_folder

        self.x_length=x_length
        self.y_length=y_length
        
        json_env = posjson_path
        with open(json_env, 'r')as fp:
            json_data = json.load(fp)
        self.z_y_x_p = json_data['z_y_x_p']
        self.tiles_dic = dict()
        for tile in self.z_y_x_p:
            self.tiles_dic[tile['Tile']]=[tile['x'],tile['y'],tile['z']]
        multiple=int(self.jsoninfo['highest_vixel_size']/self.jsoninfo['vixel_size'])
        if(real_factors<multiple):
            first_tile=False
            multiple=int(self.jsoninfo['highest_vixel_size']/self.jsoninfo['vixel_size'])
            low_res_jason_path=self.jsoninfo['output_path']+'/LR-z_y_x_p.json'
            with open(low_res_jason_path, 'r')as fp:
                low_res_json_data = json.load(fp)
            total_z_y_x_p=low_res_json_data['z_y_x_p']
            new_z_y_x_p=[]
            for tile in total_z_y_x_p:
                
                tile['x']=tile['x']*multiple
                tile['y']=tile['y']*multiple
                tile['z']=tile['z']*multiple
                
                if(tile['Tile'] in self.tiles_dic):

                    if(not first_tile):
                        offset_to_zero_z=tile['z']-self.tiles_dic[tile['Tile']][2]
                        offset_to_zero_y=tile['y']-self.tiles_dic[tile['Tile']][1]
                        offset_to_zero_x=tile['x']-self.tiles_dic[tile['Tile']][0]
                        print(offset_to_zero_z,offset_to_zero_y,offset_to_zero_x)
                        first_tile=True
                    tile['x']=offset_to_zero_x+self.tiles_dic[tile['Tile']][0]
                    tile['y']=offset_to_zero_y+self.tiles_dic[tile['Tile']][1]
                    tile['z']=offset_to_zero_z+self.tiles_dic[tile['Tile']][2]
                    self.tiles_dic[tile['Tile']][0]=tile['x']
                    self.tiles_dic[tile['Tile']][1]=tile['y']
                    self.tiles_dic[tile['Tile']][2]=tile['z']
                new_z_y_x_p.append(tile)
            self.z_y_x_p=new_z_y_x_p
        for tile in self.z_y_x_p:
            tile['x']= round(tile['x']/rescale_factor)
            tile['y']= round(tile['y']/rescale_factor)
            tile['z']= round(tile['z']/rescale_factor)
            tile['x_length']=self.x_length
            tile['y_length']=self.y_length

            
        self.tiles_num = len( self.z_y_x_p)
        self.pre = ''
        self.mid = '_'
        self.end = '.tiff'

        result_x = 0
        result_y = 0
        result_z = 0
        self.tiles = set()
        for tile in self.z_y_x_p:
            if result_x < tile['x'] + tile['x_length'] - 1:
                result_x = tile['x'] + tile['x_length'] - 1
            if result_y < tile['y'] + tile['y_length'] - 1:
                result_y = tile['y'] + tile['y_length'] - 1
            if result_z<tile['z']:
                result_z = tile['z']
            self.tiles.add(tile['Tile'])

        result_x = result_x + 1
        result_y = result_y + 1
        aaa=1
        self.planes_num = planes_num_0 + result_z
        self.result_x=result_x
        self.result_y=result_y
        self.result_z=result_z
        self.rescale_factor=int(rescale_factor)

        # 以terafly格式输出参数计算
        v_size=256
        # 分块个数
        num_y= math.ceil(result_y/v_size)
        num_x=math.ceil(result_x/v_size)
        num_z=math.ceil(self.planes_num/v_size)
        
        self.num_y=num_y
        self.num_x=num_x
        self.num_z=num_z

        # 每块的具体个数
        v_y = math.ceil(result_y/(math.ceil(result_y/v_size)))
        overstep=num_y*v_y-result_y
        num_y1=num_y-overstep
        self.y_length_list=[v_y]*num_y1+[v_y-1]*overstep
        v_x = math.ceil(result_x/(math.ceil(result_x/v_size)))
        overstep=num_x*v_x-result_x
        num_x1=num_x-overstep
        self.x_length_list=[v_x]*num_x1+[v_x-1]*overstep
        v_z = math.ceil(self.planes_num/(math.ceil(self.planes_num/v_size)))
        overstep=num_z*v_z-self.planes_num
        num_z1=num_z-overstep
        self.z_length_list=[v_z]*num_z1+[v_z-1]*overstep
        print(self.y_length_list)
        print(self.x_length_list)
        print(self.z_length_list)
        

        self.out_root_path=result_folder+'/'+"RES("+str(result_y)+'x'+str(result_x)+'x'+str(self.planes_num)+')'
        if(not(os.path.exists(self.out_root_path))):
            os.makedirs(self.out_root_path)
        print(self.z_y_x_p)
        print("准备就绪")

    def get_weght(y,x,dir):
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
    def weght_mat(self,tiles,cur_tile,num_z,y_r,x_r):
        z_length=self.planes_num_0
        
        radio_mat=np.ones((self.y_length,self.x_length),dtype=np.float16)
        y_l=int(self.y_length*y_r)
        x_l=int(self.x_length*x_r)
        for i in range(0,y_l):
            for k in range(self.x_length):
                if(k>x_l and k<self.x_length-x_l):
                    radio_mat[i,k]=i/y_l
                else:
                    radio_mat[i,k]=min(i/y_l,min((self.x_length-k)/x_l,k/x_l))
        for i in range(self.y_length-y_l,self.y_length):
            for k in range(self.x_length):
                if(k>x_l and k<self.x_length-x_l):
                    radio_mat[i,k]=(self.y_length-i)/y_l
                else:
                    radio_mat[i,k]=min((self.y_length-i)/y_l,min((self.x_length-k)/x_l,k/x_l))
        for i in range(y_l,self.y_length-y_l):
            for k in range(self.x_length):
                if(k<x_l or k>self.x_length-x_l):
                    radio_mat[i,k]=min((self.x_length-k)/x_l,k/x_l)
        # posY=int(cur_tile[1:3])
        # posX=int(cur_tile[5:7])
        # pos1=tiles.get(cur_tile)
        # up_tile="Z"+str(posY-1).zfill(2)+cur_tile[3:7]
        # down_tile='Z'+str(posY+1).zfill(2)+cur_tile[3:7]
        # left_tile=cur_tile[0:5]+str(posX+1).zfill(2)
        # right_tile=cur_tile[0:5]+str(posX-1).zfill(2)
        # LU_tile='Z'+str(posY-1).zfill(2)+"_Y"+str(posX+1).zfill(2)
        # LD_tile='Z'+str(posY+1).zfill(2)+"_Y"+str(posX+1).zfill(2)
        # RU_tile='Z'+str(posY-1).zfill(2)+"_Y"+str(posX-1).zfill(2)
        # RD_tile='Z'+str(posY+1).zfill(2)+"_Y"+str(posX-1).zfill(2)
        
        # if up_tile in tiles:
        

        #     pos2=tiles.get(up_tile)
        #     x_l=x_length-abs(pos1[0]-pos2[0])
        #     overlap_y=y_length-(pos1[1]-pos2[1])
        #     for i in range(overlap_y):
        #         if(pos1[0]<pos2[0]):
        #             radio_mat[i][-x_l:]=radio_mat[i][-x_l:]*i/overlap_y
        #         else:
        #             radio_mat[i][:x_l]=radio_mat[i][-x_l:]*i/overlap_y

        # if down_tile in tiles:

        #     pos2=tiles.get(down_tile)
        #     x_l=x_length-abs(pos1[0]-pos2[0])
        #     overlap_y=y_length-abs(pos1[1]-pos2[1])
        #     for i in range(overlap_y):
        #         if(pos1[0]<pos2[0]):
        #             radio_mat[y_length-i-1][-x_l:]=radio_mat[y_length-i-1][-x_l:]*i/overlap_y
        #         else:
        #             radio_mat[y_length-i-1][:x_l]=radio_mat[y_length-i-1][-x_l:]*i/overlap_y
                        
                    
        # if left_tile in tiles:

        #     pos2=tiles.get(left_tile)
        #     overlap_x=x_length-abs(pos1[0]-pos2[0])
        #     y_l=y_length-abs(pos1[1]-pos2[1])
        #     temp=np.zeros((overlap_x,y_l),dtype=np.float16)
        #     for i in range(overlap_x):
        #         temp[i]=i/overlap_x
        #     temp=temp.transpose(1,0)
        #     if(pos1[1]>pos2[1]):
        #         radio_mat[-y_l:,0:overlap_x]=radio_mat[-y_l:,0:overlap_x]*temp
        #     else:
        #         radio_mat[0:y_l,0:overlap_x]=radio_mat[0:y_l,0:overlap_x]*temp
                
        # if right_tile in tiles:

        #     pos2=tiles.get(right_tile)
        #     overlap_x=x_length-abs(pos1[0]-pos2[0])
        #     y_l=y_length-abs(pos1[1]-pos2[1])
        #     temp=np.zeros((overlap_x,y_l),dtype=np.float16)
        #     for i in range(overlap_x):
        #         temp[overlap_x-i-1]=i/overlap_x
        #     temp=temp.transpose(1,0)
        #     if(pos1[1]>pos2[1]):
        #         radio_mat[-y_l:,x_length-overlap_x:]=radio_mat[-y_l:,x_length-overlap_x:]*temp
        #     else:
        #         radio_mat[-y_l:,x_length-overlap_x:]=radio_mat[-y_l:,x_length-overlap_x:]*temp
                
        
        # if LU_tile in tiles:
        #     pos2=tiles.get(LU_tile)
        #     overlap_x=x_length-abs(pos1[0]-pos2[0])
        #     overlap_y=y_length-abs(pos1[1]-pos2[1])
        #     if(overlap_x>0 and overlap_y>0):
        #         radio_mat[0:overlap_y,0:overlap_x]=radio_mat[0:overlap_y,0:overlap_x]*get_weght(overlap_y,overlap_x,1)
                
        # if RU_tile in tiles:
        #     pos2=tiles.get(RU_tile)
        #     overlap_x=x_length-abs(pos1[0]-pos2[0])
        #     overlap_y=y_length-abs(pos1[1]-pos2[1])
        #     if(overlap_x>0 and overlap_y>0):
        #         radio_mat[0:overlap_y,x_length-overlap_x:]=radio_mat[0:overlap_y,x_length-overlap_x:]*get_weght(overlap_y,overlap_x,2)
                
        # if LD_tile in tiles:
        #     pos2=tiles.get(LD_tile)
        #     overlap_x=x_length-abs(pos1[0]-pos2[0])
        #     overlap_y=y_length-abs(pos1[1]-pos2[1])
        #     if(overlap_x>0 and overlap_y>0):
        #         radio_mat[y_length-overlap_y:,0:overlap_x]=radio_mat[y_length-overlap_y:,0:overlap_x]*get_weght(overlap_y,overlap_x,3)
                
        # if RD_tile in tiles:
        #     pos2=tiles.get(RD_tile)
        #     overlap_x=x_length-abs(pos1[0]-pos2[0])
        #     overlap_y=y_length-abs(pos1[1]-pos2[1])
        #     if(overlap_x>0 and overlap_y>0):
        #         radio_mat[y_length-overlap_y:,x_length-overlap_x:]=radio_mat[y_length-overlap_y:,x_length-overlap_x:]*get_weght(overlap_y,overlap_x,4)
        
        
        
        weight_stack=np.zeros((self.z_length_list[num_z],self.y_length,self.x_length),np.float16)
        for i in range(self.z_length_list[num_z]):
            weight_stack[i]=radio_mat
        return weight_stack

    def create_value_stack(z_l,y_l,x_l):
        value_stack=np.ones((z_l,y_l,x_l),dtype=np.float16)
        center_point_pos=[]
        center_point_pos.append(int(y_l/2))
        center_point_pos.append(int(x_l/2))
        value_plane=np.ones((y_l,x_l),dtype=np.float16)
        for i in range(y_l):
            for k in range(x_l):
                value1=1-1/center_point_pos[0]*abs(center_point_pos[0]-i)
                value2=1-1/center_point_pos[1]*abs(center_point_pos[1]-k)
                value_plane[i][k]=max(value1,value2)
        for z in range(z_l):
            value_stack[z]=value_stack[z]*value_plane
        return value_stack
        
        
    # value_stack=create_value_stack(planes_num_0,y_length,x_length)

        

    def create_mdata(self,vsize=256):
        # 以terafly格式输出参数计算
            
        v_size=256
        # 分块个数

        second_dir=os.listdir(self.out_root_path)
        i=0
        while(i<second_dir.__len__()):
            for k in range(second_dir[i].__len__()):
                if second_dir[i][k] =='.':
                    del second_dir[i]
                    break
                if(k==second_dir[i].__len__()-1):
                    if second_dir[i][k] !='.':
                        i=i+1
                    


        ans=struct.unpack('f', b'\x00\x00\x80\x3F')
        with open(self.out_root_path+'/'+"mdata"+".bin", "wb") as f:
            # version
            version=2.
            f.write( struct.pack('<f', version))
            # 参考坐标
            f.write(struct.pack('<i', 1))
            f.write(struct.pack('<i', 2))
            f.write(struct.pack('<i', 3))
            # 降采样倍数
            size=float(self.real_factors)
            f.write(struct.pack('<f', size)*6)
            # 原点坐标
            ori_pos=0
            f.write(struct.pack('<i',ori_pos )*3)
            # 整体尺寸
            VXL_V=self.result_y
            VXL_H=self.result_x
            VXL_D=self.planes_num
            f.write(struct.pack('<l',VXL_V ))
            f.write(struct.pack('<l',VXL_H) )
            f.write(struct.pack('<l',VXL_D ))
            # 纵向横向的分块次数
            N_ROWS=len(second_dir)  
            N_COLS=len(os.listdir(self.out_root_path+'/'+second_dir[0]))
            f.write(struct.pack('<H',N_ROWS ))
            f.write(struct.pack('<H',N_COLS))
            print(N_ROWS,N_COLS)
            # 
            for r in range(N_ROWS):
                for c in tqdm(range (N_COLS)):
                    print(r,c,self.out_root_path+'/'+second_dir[r])
                    third_dir=(os.listdir(self.out_root_path+'/'+second_dir[r]))[c]
                    cur_path=self.out_root_path+"/"+second_dir[r]+'/'+third_dir
                    file_name=os.listdir(cur_path)
                    # 最小单位下的各个尺寸
                    HEIGHT=self.y_length_list[r]
                    WIDTH=self.x_length_list[c]
                    DEPTH=VXL_D
                    f.write(struct.pack('<i',HEIGHT ))
                    f.write(struct.pack('<i',WIDTH))
                    f.write(struct.pack('<i',DEPTH))
                    DEPTH_num=len(file_name)   
                    # depth上的分块数
                    f.write(struct.pack('<i',DEPTH_num))
                    # 开始符 
                    f.write(struct.pack('<i',1))
                    # 该二级目录对应的坐标
                    if(r>0):
                        posY=np.sum(self.y_length_list[:r])
                    else:
                        posY=0
                    if(c>0):
                        posX=np.sum(self.x_length_list[:c])
                    else:
                        posX=0
                    f.write(struct.pack('<i',posY))
                    f.write(struct.pack('<i',posX))
                    # third_path 长度
                    third_path=second_dir[r]+'/'+third_dir
                    str_size=len(third_path)+1
                    f.write(struct.pack('<H',str_size))
                    # 目录名
                    
                    DIR_NAME=bytes(third_path, encoding = "utf8") 
                    f.write(DIR_NAME)
                    f.write(struct.pack('B',0))
                    # 文件名
                    for t in range(len(file_name)):
                        file_name_size=len(file_name[t])+1
                        f.write(struct.pack('<H',file_name_size))
                        FILE_NAME=bytes(file_name[t], encoding = "utf8") 
                        f.write(FILE_NAME)
                        f.write(struct.pack('B',0))
                        # 该块的深度和深度坐标
                        cur_depth=self.z_length_list[t]
                        posZ=0
                        if(t>0):
                            posZ=posX=np.sum(self.z_length_list[:t])
                        f.write(struct.pack('<i',cur_depth))
                        f.write(struct.pack('<i',posZ))
                    # 结束符号
                    f.write(struct.pack('<i',2))
            
            
    # stitch part
    def work(self,i,k,j):
            Normalization_stack= np.zeros((self.z_length_list[j],self.y_length_list[i], self.x_length_list[k]),dtype=np.float16)
           
            cur_tera_stack=np.zeros((self.z_length_list[j],self.y_length_list[i], self.x_length_list[k]),dtype=np.float16)
            l_ = len(str(self.tiles_num))
            tile_path_list = os.listdir(self.input_folder)
            #print(tiles_num)
            if(i==0):
                posY=0
            else:
                posY=np.sum(self.y_length_list[:i])

            if(k==0):
                posX=0
            else:
                posX=np.sum(self.x_length_list[:k]) 
                    
            if(j==0):
                posZ=0
            else:
                posZ=np.sum(self.z_length_list[:j])
            str_y=str(posY*10).zfill(6)
            str_x=str(posX*10).zfill(6)    
            str_z=str(posZ*10).zfill(6) 
            out_path=self.out_root_path+'/'+str_y+'/'+str_y+'_'+str_x
            file_name=str_y+'_'+str_x+'_'+str_z+'.tif'
            if(os.path.exists(out_path+'/'+file_name)):
                return

            for m in range(self.tiles_num):

                temp_t = self.z_y_x_p[m]
                l_str = temp_t['Tile']
                offset_y=int(temp_t['y']-posY)
                offset_x=int(temp_t['x']-posX)
                offset_z=int(temp_t['z'])
                                     
                if(not os.path.exists(self.input_folder+'/'+l_str+'.tiff')):
                    continue
                if(self.mode==2 and self.jsoninfo['high_res_location']=='None'):
                    continue
                if (posY+self.y_length_list[i])<= temp_t['y'] or posY >= (temp_t['y']+self.y_length) or (posX+self.x_length_list[k])<= temp_t['x'] or posX>=(temp_t['x']+self.x_length) or posZ>=(offset_z+self.planes_num_0) or (posZ+self.z_length_list[j])<=offset_z:
                    continue

                print(temp_t['Tile'],i,k)

                image_stack=np.zeros((self.z_length_list[j],self.y_length, self.x_length),dtype=np.uint16)
               
                     

                img_name=self.input_folder+'/'+l_str+'.tiff'

                with TiffFile(img_name) as tiff_img:
                # Read only the first image from the file
                    if(offset_z<=posZ):#只截取下部分
                        if(offset_z+self.planes_num_0>=posZ+self.z_length_list[j]):#整个空间
                            t_stack=tiff_img.asarray(key=slice(((posZ-offset_z))*self.rescale_factor,(posZ-offset_z+self.z_length_list[j])*self.rescale_factor))
                            sp=t_stack.shape
                            if(len(sp)==2):
                                rescale_factors=(self.rescale_factor,self.rescale_factor)
                            else:
                                rescale_factors=(self.rescale_factor,self.rescale_factor,self.rescale_factor)
                            resized_t_stack = downsample_array(t_stack, rescale_factors)
                            image_stack= resized_t_stack
                            
                        else:
                            t_stack= tiff_img.asarray(key=slice((posZ-offset_z)*self.rescale_factor,(self.planes_num_0)*self.rescale_factor))
                            sp=t_stack.shape
                            if(len(sp)==2):
                                rescale_factors=(self.rescale_factor,self.rescale_factor)
                            else:
                                rescale_factors=(self.rescale_factor,self.rescale_factor,self.rescale_factor)
                            resized_t_stack = downsample_array(t_stack, rescale_factors)        
                            image_stack[:(self.planes_num_0-posZ+offset_z),:,:]= resized_t_stack
                    else:
                        if(offset_z+self.planes_num_0>=posZ+self.z_length_list[j]):
                            t_stack= tiff_img.asarray(key=slice(0,(posZ-offset_z+self.z_length_list[j])*self.rescale_factor))
                            sp=t_stack.shape
                            if(len(sp)==2):
                                rescale_factors=(self.rescale_factor,self.rescale_factor)
                            else:
                                rescale_factors=(self.rescale_factor,self.rescale_factor,self.rescale_factor)
                            resized_t_stack = downsample_array(t_stack, rescale_factors)
                            image_stack[-(posZ-offset_z+self.z_length_list[j]):,:,:] = resized_t_stack
                        else:
                            t_stack= tiff_img.asarray(key=slice(0,(self.planes_num_0))*self.rescale_factor)
                            sp=t_stack.shape
                            if(len(sp)==2):
                                rescale_factors=(self.rescale_factor,self.rescale_factor)
                            else:
                                rescale_factors=(self.rescale_factor,self.rescale_factor,self.rescale_factor)
                            resized_t_stack = downsample_array(t_stack, rescale_factors)
                            image_stack[offset_z-posZ:offset_z-posZ+self.planes_num_0] =resized_t_stack
                            
                            
                        
                        
                    
                image_stack=image_stack.astype(np.float16)
                weight_stack=self.weght_mat(tiles=self.tiles_dic,cur_tile=l_str,num_z=j,y_r=self.jsoninfo['overlapY'],x_r=self.jsoninfo['overlapX'])
                image_stack=image_stack*weight_stack


                if(offset_y<0):
                    if(offset_x<0):
                        t_y=min(offset_y+self.y_length,self.y_length_list[i])
                        t_x=min(offset_x+self.x_length,self.x_length_list[k])
                        cur_tera_stack[:,0:t_y,0:t_x]+=image_stack[:,-t_y:,-t_x:]
                        Normalization_stack[:,0:t_y,0:t_x]+=weight_stack[:,-t_y:,-t_x:]
                    else:
                        t_y=min(offset_y+self.y_length,self.y_length_list[i])
                        t_x=min(self.x_length_list[k]-offset_x,min(self.x_length_list[k],self.x_length))
                        cur_tera_stack[:,0:t_y,offset_x:offset_x+t_x]+=image_stack[:,-t_y:,0:t_x]
                        Normalization_stack[:,0:t_y,offset_x:offset_x+t_x]+=weight_stack[:,-t_y:,0:t_x]
                else:
                    if(offset_x<0):
                        t_y=min(self.y_length_list[i]-offset_y,min(self.y_length,self.y_length_list[i]))
                        t_x=min(offset_x+self.x_length,self.x_length_list[k])
                        cur_tera_stack[:,offset_y:offset_y+t_y,0:t_x]+=image_stack[:,0:t_y,-t_x:]
                        Normalization_stack[:,offset_y:offset_y+t_y,0:t_x]+=weight_stack[:,0:t_y,-t_x:]
                    else:
                        t_y=min(self.y_length_list[i]-offset_y,min(self.y_length,self.y_length_list[i])) 
                        t_x=min(self.x_length_list[k]-offset_x,min(self.x_length_list[k],self.x_length))
                        cur_tera_stack[:,offset_y:offset_y+t_y,offset_x:offset_x+t_x]+=image_stack[:,0:t_y,0:t_x]
                        Normalization_stack[:,offset_y:offset_y+t_y,offset_x:offset_x+t_x]+=weight_stack[:,0:t_y,0:t_x]

            if(not(os.path.exists(out_path))):
                os.makedirs(out_path)  
            # print(file_name,stack_totalz.shape)
            cur_tera_stack/=Normalization_stack
            cur_tera_stack=cur_tera_stack.astype(np.uint16)
            tiff.imwrite(out_path+'/'+file_name,cur_tera_stack,compression=5)
            print(file_name)

    def run(self):
        process_num=int(multiprocessing.cpu_count()/2)
        pool = multiprocessing.Pool(processes=process_num)
        for i in tqdm(range(self.num_y)):
            for k in tqdm(range(self.num_x)):
                for j in tqdm(range(self.num_z)):
                    pool.apply_async(self.work, args=(i,k,j,))
                    # pool.apply_async(func=work,args=(i,k,j,))
                    # self.work(i,k,j)
        pool.close()
        pool.join()   
        print("start to create mdata file...")
        self.create_mdata(256)
        
        

