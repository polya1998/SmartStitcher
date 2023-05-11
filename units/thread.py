# -*- coding: utf-8 -*-
import time
from PyQt5.QtCore import QThread, pyqtSignal
import JsonInfo
from units.json_tools import *
import os
import numpy as np
import tifffile
from units.image_stitch_tool import*
from units.shift_compute import *
from units.global_optimazation import *
from units.float2int_tool import*
from units.MIP_stitched import *
from units.z_slice import *
from units.make_slice import *
from units.output_terafly import *

class MIP_Thread(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self, data,input_path,second_path,locations,parent=None):
        super(MIP_Thread, self).__init__(parent)
        self.second_path=second_path
        self.data = data
        self.input_path=input_path
        self.locations=locations
        self.progress=0
    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):
        print("start mip")
        mip_path=self.second_path
        # if 'MIP_dir'in self.data:
        #     self.finishSignal.emit(100)
        #     return
        if  self.input_path=="":
            self.finishSignal.emit(-1)
            return
        mip_path=self.data['output_path']+'/'+mip_path
        mip_path_z=mip_path+'/Z_MIP'
        mip_path_x=mip_path+'/X_MIP'
        mip_path_y=mip_path+'/Y_MIP'
        print(mip_path)
        if(not os.path.exists(mip_path_z)):
            os.makedirs(mip_path_z)
        if(not os.path.exists(mip_path_x)):
            os.makedirs(mip_path_x)
        if(not os.path.exists(mip_path_y)):
            os.makedirs(mip_path_y)
        folder_path=self.input_path
        file_names = [f for f in os.listdir(folder_path) if (f.endswith('.tif') or f.endswith('.tiff'))]
        # 遍历每个tiff图像，计算x、y、z三个方向的MIP，并保存结果为新的tiff文件
        num_file=len(self.locations)*len(self.locations[0])
        cur_num=0
        process_num=int(10)
        pool = multiprocessing.Pool(processes=process_num)
        for line in self.locations:
            for file_name in line:
                # 构造新文件名
                if(file_name!='None'):
                    print(file_name)
                    
                    mip_x_file_name = file_name+ '_mip_x.tiff'
                    mip_y_file_name = file_name+ '_mip_y.tiff'
                    mip_z_file_name = file_name+ '_mip_z.tiff'
                    # 计算MIPmip_get并保存为新的tiff文件
                    # pool.apply_async(self.mip_get, args=(folder_path,file_name,mip_x_file_name,mip_path_x,2,cur_num,num_file))
                    # pool.apply_async(self.mip_get, args=(folder_path,file_name,mip_y_file_name,mip_path_y,1,cur_num,num_file))
                    # pool.apply_async(self.mip_get, args=(folder_path,file_name,mip_z_file_name,mip_path_z,0,cur_num,num_file))
                    self.mip_get(folder_path,file_name,mip_x_file_name,mip_path_x,2,cur_num,num_file)
                    self.mip_get(folder_path,file_name,mip_y_file_name,mip_path_y,1,cur_num,num_file)
                    self.mip_get(folder_path,file_name,mip_z_file_name,mip_path_z,0,cur_num,num_file)
                cur_num+=1
        # pool.close()
        # pool.join()  
        self.finishSignal.emit(100)


    def mip_get(self,folder_path,file_name,output_name,save_path,axis,count,num_file):
            print(file_name)
            if(not os.path.exists(os.path.join(save_path, output_name))):
                mip_x = compute_mip(os.path.join(folder_path, file_name+'.tiff'), axis)
                tifffile.imwrite(os.path.join(save_path, output_name), mip_x)
            progress=int(count/num_file*100)
            self.progress=max(self.progress,progress)
            self.finishSignal.emit(self.progress)
            
                
            
class shiftcompute_Thread(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self, data,input_folder,location,outputname,parent=None):
        super(shiftcompute_Thread, self).__init__(parent)
        self.json_data = data
        self.input_folder=input_folder
        self.location=location
        self.outputname=outputname
    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):
        print(self.json_data)
        input_folder =  self.input_folder+'/Z_MIP'
        locations = self.location
        namelisft_Z_mip=os.listdir(input_folder)
        sp=tiff.imread( input_folder+'/' + namelisft_Z_mip[0]).shape
        # tiles_names = json_data['tiles_names']
        x_length = int(sp[1])
        y_length = int(sp[0])
        overlap_x=self.json_data['overlapX']
        overlap_y=self.json_data['overlapY']
        x_shift_d_ = int(math.ceil(x_length*0.02))
        y_shift_d = int(math.ceil(y_length*overlap_y))    
        y_range = int(math.ceil(y_length*overlap_y*1.3))
        y_shift_d_ = int(math.ceil(x_length*0.02))
        x_shift_d = int(math.ceil(x_length*overlap_x))    
        x_range = int(math.ceil(x_length*overlap_x*1.3))
        output_folder=self.json_data['output_path']
        print(output_folder)
        if(os.path.exists(output_folder+'/'+self.outputname)):
            self.finishSignal.emit(100)
            return
        locations = np.asarray(locations)
        res_l = []
        for i in tqdm(range(locations.shape[0])):
            for j in range(locations.shape[1]):
                if locations[i][j] != 'None':
                    if i<locations.shape[0]-1 and locations[i+1][j]!= 'None':
                        res = xy_get_shift(locations[i][j],
                                                            locations[i+1][j],
                                                            input_folder+'/' + locations[i][j]+'_mip_z.tiff' ,
                                                            input_folder+'/' + locations[i+1][j]+'_mip_z.tiff',
                                                            0,
                                                            x_shift_d_,
                                                            y_shift_d,
                                                            y_range
                                                            )
                        res_l.append(res)
                    if j < locations.shape[1] - 1 and locations[i][j+1] != 'None':
                        res = xy_get_shift(locations[i][j],
                                                            locations[i][j+1],
                                                            input_folder + '/' + locations[i][j]+'_mip_z.tiff' ,
                                                            input_folder + '/' + locations[i][j+1]+'_mip_z.tiff',
                                                            1,
                                                            y_shift_d_,
                                                            x_shift_d,
                                                            x_range
                                                            )
                        res_l.append(res)
                process=((i)*(locations.shape[1])+j+1)/(locations.shape[1]*locations.shape[0])*100
                print(process)
                self.finishSignal.emit(process)

        result_y_x_s = []
        for res in res_l:
            result_y_x_s.append(res)
        print('write json file')
        result = {#'result_z_s': result_z_s,
                        'result_y_x_s': result_y_x_s
                }
        open(output_folder+'/'+self.outputname, 'w')
        with open(output_folder+'/'+self.outputname, 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


class Z_shift_Thread(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self, data,input_folder,location,outputname,parent=None):
        super(Z_shift_Thread, self).__init__(parent)
        self.json_data = data
        self.input_folder=input_folder
        self.location=location
        self.outputname=outputname
    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):
        # print(json_data)
        input_folder = self.input_folder
        namelisft_X_mip=os.listdir(input_folder + '/X_MIP')
        namelisft_Y_mip=os.listdir(input_folder + '/Y_MIP')
        locations = self.location
        sp=tiff.imread(input_folder + '/X_MIP/' + namelisft_X_mip[0]).shape
        y_length = int(sp[1])
        z_length = int(sp[0])
        sp=tiff.imread(input_folder + '/Y_MIP/' + namelisft_Y_mip[1]).shape
        # tiles_names = json_data['tiles_names']
        x_length = int(sp[1])
        overlap_x=self.json_data['overlapX']
        overlap_y=self.json_data['overlapY']
        x_shift_d_ = int(math.ceil(x_length*0.02))
        y_shift_d = int(math.ceil(y_length*overlap_y))    
        y_range = int(math.ceil(y_length*overlap_y*1.3))
        y_shift_d_ = int(math.ceil(x_length*0.02))
        x_shift_d = int(math.ceil(x_length*overlap_x))    
        x_range = int(math.ceil(x_length*overlap_x*1.3))
        output_folder=self.json_data['output_path']
        z_shift_d=int(math.ceil(z_length*self.json_data['overlapZ']))
        if(os.path.exists(output_folder+'/'+self.outputname)):
            self.finishSignal.emit(100)
            return
        res_l = []
        locations = np.asarray(locations)

        for i in tqdm(range(locations.shape[0])):
            for j in range(locations.shape[1]):
                if locations[i][j] != 'None':
                    if i<locations.shape[0]-1 and locations[i+1][j]!= 'None':
                        
                        res = z_get_shift(locations[i][j],
                                                            locations[i+1][j],
                                                            input_folder + '/X_MIP/' + locations[i][j]+'_mip_x.tiff' ,
                                                            input_folder + '/X_MIP/' +locations[i+1][j]+'_mip_x.tiff' ,
                                                            0,
                                                            z_shift_d,
                                                            math.ceil(y_shift_d),
                                                            math.ceil(y_shift_d*1.3)
                                                            ,
                                                            z_length
                                                            )
                        res_l.append(res)
                    if j < locations.shape[1] - 1 and locations[i][j+1] != 'None':
                        res = z_get_shift(locations[i][j],
                                                            locations[i][j+1],
                                                            input_folder + '/Y_MIP/' + locations[i][j]+'_mip_y.tiff'  ,
                                                           input_folder + '/Y_MIP/' + locations[i][j+1]+'_mip_y.tiff' ,
                                                            1,
                                                            z_shift_d,
                                                             math.ceil(x_shift_d),
                                                            math.ceil(x_shift_d*1.3)
                                                            ,
                                                            z_length
                                                            )
                        res_l.append(res)
                process=((i)*(locations.shape[1])+j+1)/(locations.shape[1]*locations.shape[0])*100
                print(process)
                self.finishSignal.emit(process)
        result_z_yx_s = []
        for res in res_l:
            result_z_yx_s.append(res)
        print('write json file')
        result = {#'result_z_s': result_z_s,
                        'result_z_yx_s': result_z_yx_s
                }
        open(output_folder+'/'+self.outputname, 'w')
        with open(output_folder+'/'+self.outputname, 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        


        
def compute_mip(file_path, axis):
    # 读取tiff图像
    img = tifffile.imread(file_path)
    # 将图像转换为NumPy数组
    img_array = np.array(img)
    # 计算MIP
    mip = np.max(img_array, axis=axis)
    return mip


class gloabal_opi_Thread(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self,data,isHighres,parent=None):
        super(gloabal_opi_Thread, self).__init__(parent)
        self.jsondata = data
        self.isHighres=isHighres
    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):
        t_start=time.perf_counter()
        if( not self.isHighres):
            apply_highres_offsets_to_lowres_G_O_Z(self.jsondata['output_path']+'/HR-Z-shift_manual_g_o_position.json',self.jsondata['output_path']+"/low_res_z_shift.json",self.jsondata['output_path']+"/LR-Z-shift_manual_g_o_position.json",int(self.jsondata['highest_vixel_size']/self.jsondata['vixel_size']))
        else :
            G_O_Z(self.jsondata['output_path']+"/high_res_z_shift.json",self.jsondata['output_path']+"/HR-Z-shift_manual_g_o_position.json")
        t_now=time.perf_counter()-t_start

        self.finishSignal.emit(33)
        if( not self.isHighres):
            apply_highres_offsets_to_lowres_G_O_Y(self.jsondata['output_path']+'/HR-Y-shift_manual_g_o_position.json',self.jsondata['output_path']+"/low_res_x_y_shift.json",self.jsondata['output_path']+"/LR-Y-shift_manual_g_o_position.json",int(self.jsondata['highest_vixel_size']/self.jsondata['vixel_size']))
        else:
            G_O_Y(self.jsondata['output_path']+"/high_res_x_y_shift.json",self.jsondata['output_path']+"/HR-Y-shift_manual_g_o_position.json")

        self.finishSignal.emit(66)
        if( not self.isHighres):
            apply_highres_offsets_to_lowres_G_O_X(self.jsondata['output_path']+'/HR-X-shift_manual_g_o_position.json',self.jsondata['output_path']+"/low_res_x_y_shift.json",self.jsondata['output_path']+"/LR-X-shift_manual_g_o_position.json",int(self.jsondata['highest_vixel_size']/self.jsondata['vixel_size']))
        else:    
            G_O_X(self.jsondata['output_path']+"/high_res_x_y_shift.json",self.jsondata['output_path']+"/HR-X-shift_manual_g_o_position.json")

        self.finishSignal.emit(100)
        
class float2int_Thread(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self,data,isHigh,parent=None):
        super(float2int_Thread, self).__init__(parent)
        self.jsoninfo = data
        self.isHigh=isHigh
    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):
        if( self.isHigh):
            zz(self.jsoninfo['output_path']+"/HR-Z-shift_manual_g_o_position.json",self.jsoninfo['output_path']+"/HR-Z-shift_manual_g_o_position_z.json")
            self.finishSignal.emit(25)
            yy(self.jsoninfo['output_path']+"/HR-Y-shift_manual_g_o_position.json",self.jsoninfo['output_path']+"/HR-Y-shift_manual_g_o_position_y.json")
            self.finishSignal.emit(50)
            xx(self.jsoninfo['output_path']+"/HR-X-shift_manual_g_o_position.json",self.jsoninfo['output_path']+"/HR-X-shift_manual_g_o_position_x.json")
            self.finishSignal.emit(75)
            mip_folder=self.jsoninfo['output_path']+'/high_res_MIP/Z_MIP'
            namelisft_Z_mip=os.listdir(mip_folder)
            sp=tiff.imread( mip_folder+'/' + namelisft_Z_mip[0]).shape
            write_z_y_x(self.jsoninfo['output_path'],self.jsoninfo['output_path']+"/HR-z_y_x_p.json",sp[1],sp[0],self.isHigh)
            self.finishSignal.emit(100)
        else:
            zz(self.jsoninfo['output_path']+"/LR-Z-shift_manual_g_o_position.json",self.jsoninfo['output_path']+"/LR-Z-shift_manual_g_o_position_z.json")
            self.finishSignal.emit(25)
            yy(self.jsoninfo['output_path']+"/LR-Y-shift_manual_g_o_position.json",self.jsoninfo['output_path']+"/LR-Y-shift_manual_g_o_position_y.json")
            self.finishSignal.emit(50)
            xx(self.jsoninfo['output_path']+"/LR-X-shift_manual_g_o_position.json",self.jsoninfo['output_path']+"/LR-X-shift_manual_g_o_position_x.json")
            self.finishSignal.emit(75)
            mip_folder=self.jsoninfo['output_path']+'/low_res_MIP/Z_MIP'
            namelisft_Z_mip=os.listdir(mip_folder)
            sp=tiff.imread( mip_folder+'/' + namelisft_Z_mip[0]).shape
            write_z_y_x(self.jsoninfo['output_path'],self.jsoninfo['output_path']+"/LR-z_y_x_p.json",sp[1],sp[0],self.isHigh)
            self.finishSignal.emit(100)
            
class make_LR_MIP_stitched(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self,data,parent=None):
        super(make_LR_MIP_stitched, self).__init__(parent)
        self.jsoninfo=data

        

    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):       
        json_env =self.jsoninfo['output_path']+ '/LR-z_y_x_p.json'

        with open(json_env, 'r')as fp:
            json_data = json.load(fp)
        z_y_x_p = json_data['z_y_x_p']

        tiles_dic = dict()
        for tile in z_y_x_p:
            tiles_dic[tile['Tile']]=[tile['x'],tile['y']]
        print(tiles_dic)

        input_folder = self.jsoninfo['output_path']+'/low_res_MIP/Z_MIP'
        result_folder = self.jsoninfo['output_path']+'/LR_result_slice'
        if(not(os.path.exists(result_folder))):
            os.makedirs(result_folder)
        mip_path=self.jsoninfo['output_path']+'/low_res_MIP/X_MIP'
        namelisft_X_mip=os.listdir(mip_path )
        sp=tiff.imread(mip_path+'/'+namelisft_X_mip[0]).shape
        planes_num =sp[0] 
        thread_num = 10
        planes_num_0 =planes_num
        tiles_num = len(z_y_x_p)
        pre = ''
        mid = '_'
        end = '.tiff'
        result_x = 0
        result_y = 0
        result_z = 0
        tiles = set()
        for tile in z_y_x_p:
            if result_x < tile['x'] + tile['x_length'] - 1:
                result_x = tile['x'] + tile['x_length'] - 1
            if result_y < tile['y'] + tile['y_length'] - 1:
                result_y = tile['y'] + tile['y_length'] - 1
            if result_z<tile['z']:
                result_z = tile['z']
            tiles.add(tile['Tile'])

        result_x = result_x + 1
        result_y = result_y + 1
        planes_num = planes_num + result_z
        aaa=1
        out_zslice_part=MIP_stitched_c(planes_num_0,z_y_x_p,planes_num,result_x,result_y,result_folder
                 ,tiles_num,input_folder,pre,mid,end)
        out_zslice_part.work()   
        self.finishSignal.emit(100)  
        
        
class make_HR_MIP_stitched(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self,data,parent=None):
        super(make_HR_MIP_stitched, self).__init__(parent)
        self.jsoninfo=data

        

    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):       
        json_env =self.jsoninfo['output_path']+ '/HR-z_y_x_p.json'

        with open(json_env, 'r')as fp:
            json_data = json.load(fp)
        z_y_x_p = json_data['z_y_x_p']

        tiles_dic = dict()
        for tile in z_y_x_p:
            tiles_dic[tile['Tile']]=[tile['x'],tile['y']]
        print(tiles_dic)

        input_folder = self.jsoninfo['output_path']+'/high_res_MIP/Z_MIP'
        result_folder = self.jsoninfo['output_path']+'/HR_result_slice'
        if(not(os.path.exists(result_folder))):
            os.makedirs(result_folder)
        mip_path=self.jsoninfo['output_path']+'/high_res_MIP/X_MIP'
        namelisft_X_mip=os.listdir(mip_path )
        sp=tiff.imread(mip_path+'/'+namelisft_X_mip[0]).shape
        planes_num =sp[0] 
        thread_num = 10
        planes_num_0 =planes_num
        tiles_num = len(z_y_x_p)

        print(tiles_num)
        pre = ''
        mid = '_'
        end = '.tiff'
        result_x = 0
        result_y = 0
        result_z = 0
        tiles = set()
        for tile in z_y_x_p:
            if result_x < tile['x'] + tile['x_length'] - 1:
                result_x = tile['x'] + tile['x_length'] - 1
            if result_y < tile['y'] + tile['y_length'] - 1:
                result_y = tile['y'] + tile['y_length'] - 1
            if result_z<tile['z']:
                result_z = tile['z']
            tiles.add(tile['Tile'])

        result_x = result_x + 1
        result_y = result_y + 1
        planes_num = planes_num + result_z
        aaa=1
        out_zslice_part=MIP_stitched_c(planes_num_0,z_y_x_p,planes_num,result_x,result_y,result_folder
                 ,tiles_num,input_folder,pre,mid,end)
        out_zslice_part.work()   
        self.finishSignal.emit(100)
        
  
  
class LR_make_slice(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self,data,parent=None):
        super(LR_make_slice, self).__init__(parent)
        self.jsoninfo=data

        

    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):       
        json_env =self.jsoninfo['output_path']+ '/LR-z_y_x_p.json'

        with open(json_env, 'r')as fp:
            json_data = json.load(fp)
        z_y_x_p = json_data['z_y_x_p']

        tiles_dic = dict()
        for tile in z_y_x_p:
            tiles_dic[tile['Tile']]=[tile['x'],tile['y']]
        print(tiles_dic)

        input_folder = self.jsoninfo['input_path']
        result_folder = self.jsoninfo['output_path']+'/LR_result_slice'
        if(not(os.path.exists(result_folder))):
            os.makedirs(result_folder)
        mip_path=self.jsoninfo['output_path']+'/low_res_MIP/X_MIP'
        namelisft_X_mip=os.listdir(mip_path )
        sp=tiff.imread(mip_path+'/'+namelisft_X_mip[0]).shape
        planes_num =sp[0] 
        thread_num = 10
        planes_num_0 =planes_num
        tiles_num = len(z_y_x_p)

        print(tiles_num)
        pre = ''
        mid = '_'
        end = '.tiff'
        result_x = 0
        result_y = 0
        result_z = 0
        tiles = set()
        for tile in z_y_x_p:
            if result_x < tile['x'] + tile['x_length'] - 1:
                result_x = tile['x'] + tile['x_length'] - 1
            if result_y < tile['y'] + tile['y_length'] - 1:
                result_y = tile['y'] + tile['y_length'] - 1
            if result_z<tile['z']:
                result_z = tile['z']
            tiles.add(tile['Tile'])

        result_x = result_x + 1
        result_y = result_y + 1
        planes_num = planes_num + result_z
        aaa=1
        out_zslice_part=z_slice_c(planes_num_0,z_y_x_p,planes_num,result_x,result_y,result_folder
                 ,tiles_num,input_folder,pre,mid,end)
        out_zslice_part.work()
        self.finishSignal.emit(100)  
        
class HR_make_slice(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self,data,parent=None):
        super(HR_make_slice, self).__init__(parent)
        self.jsoninfo=data

        

    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):       
        json_env =self.jsoninfo['output_path']+ '/HR-z_y_x_p.json'

        with open(json_env, 'r')as fp:
            json_data = json.load(fp)
        z_y_x_p = json_data['z_y_x_p']

        tiles_dic = dict()
        for tile in z_y_x_p:
            tiles_dic[tile['Tile']]=[tile['x'],tile['y']]
        print(tiles_dic)

        input_folder = self.jsoninfo['highest_path']
        result_folder = self.jsoninfo['output_path']+'/HR_result_slice'
        if(not(os.path.exists(result_folder))):
            os.makedirs(result_folder)
        mip_path=self.jsoninfo['output_path']+'/high_res_MIP/X_MIP'
        namelisft_X_mip=os.listdir(mip_path )
        sp=tiff.imread(mip_path+'/'+namelisft_X_mip[0]).shape
        planes_num =sp[0] 
        thread_num = 10
        planes_num_0 =planes_num
        tiles_num = len(z_y_x_p)

        print(tiles_num)
        pre = ''
        mid = '_'
        end = '.tiff'
        result_x = 0
        result_y = 0
        result_z = 0
        tiles = set()
        for tile in z_y_x_p:
            if result_x < tile['x'] + tile['x_length'] - 1:
                result_x = tile['x'] + tile['x_length'] - 1
            if result_y < tile['y'] + tile['y_length'] - 1:
                result_y = tile['y'] + tile['y_length'] - 1
            if result_z<tile['z']:
                result_z = tile['z']
            tiles.add(tile['Tile'])

        result_x = result_x + 1
        result_y = result_y + 1
        planes_num = planes_num + result_z
        aaa=1
        out_zslice_part=z_slice_c(planes_num_0,z_y_x_p,planes_num,result_x,result_y,result_folder
                 ,tiles_num,input_folder,pre,mid,end)
        out_zslice_part.work()
        self.finishSignal.emit(100)


class make_slice(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self,locations,input_folder,z_length,save_dirpath,parent=None):
        super(make_slice, self).__init__(parent)
        self.json_data=data
        self.input_folder=input_folder
        self.z_length=z_length
        self.save_dirpath=save_dirpath
        self.locations=locations
        
    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):       
        input_folder = self.input_folder
        locations = self.locations
        locations = np.asarray(locations)
        sum_num=locations.shape[0]*locations.shape[1]
        cur=0
        for n in tqdm(range(0,locations.shape[0],1)):
            for m in range(0,locations.shape[1],1):
                    cur+=1
                    print(n,m)
                    name=locations[n][m]
                    if(name!='None'):
                        print(name)
                        cur_tile=tiff.imread(input_folder+'/'+name+'.tiff')
                        sp=cur_tile.shape
                        process_num=10
                        step=int(self.z_length/process_num)
                        process_list = []
                        # print(name,sp,step)
                        for i in range(0,sp[0],step):  #开启5个子进程执行fun1函数
                            cur_tile_part=cur_tile[i:i+step]
                            p = Process(target=work,args=(cur_tile_part,name,self.save_dirpath,i,self.z_length,)) #实例化进程对象
                            p.start()
                            process_list.append(p)
                        for i in process_list:
                            p.join()
                        del process_list
                        del cur_tile
                        print("完成！"+name)
                    t=int(cur/sum_num)*100
                    self.finishSignal.emit(t)


class terafly_output_thread(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self,data,parent=None):
        super(terafly_output_thread, self).__init__(parent)
        self.jsoninfo=data


        

    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):  
        multiple=int(self.jsoninfo['highest_vixel_size']/self.jsoninfo['vixel_size'])
        
        for cur_res in range(0,6,1):
            if(pow(2,cur_res)<multiple):
                if(self.jsoninfo['res_list'][str(cur_res)]['part_atitch']):
                    x = self.jsoninfo['res_list'][str(cur_res)]['res'].split("x")
                    out_terafly_part=output_terafly_c(data=self.jsoninfo,posjson_path=self.jsoninfo['output_path']+'/HR-z_y_x_p.json',input_folder=self.jsoninfo['highest_path'],
                                                     result_folder=self.jsoninfo['output_path']+'/result_terafly',x_length=int(x[0]),y_length=int(x[1]),planes_num_0=int(x[2]),rescale_factor=pow(2,cur_res),real_factors=pow(2,cur_res),mode=2)
                    out_terafly_part.run()
                if(self.jsoninfo['res_list'][str(cur_res)]['all_stitch']):
                    x = self.jsoninfo['res_list'][str(cur_res)]['res'].split("x")
                    out_terafly_part=output_terafly_c(data=self.jsoninfo,posjson_path=self.jsoninfo['output_path']+'/HR-z_y_x_p.json',input_folder=self.jsoninfo['highest_path'],
                                                     result_folder=self.jsoninfo['output_path']+'/result_terafly',x_length=int(x[0]),y_length=int(x[1]),planes_num_0=int(x[2]),rescale_factor=pow(2,cur_res),real_factors=pow(2,cur_res),mode=1)
                    out_terafly_part.run()
                    
            else:
                if(self.jsoninfo['res_list'][str(cur_res)]['all_stitch']):
                    x = self.jsoninfo['res_list'][str(cur_res)]['res'].split("x")
                    out_terafly_part=output_terafly_c(data=self.jsoninfo,posjson_path=self.jsoninfo['output_path']+'/LR-z_y_x_p.json',input_folder=self.jsoninfo['input_path'],
                                                     result_folder=self.jsoninfo['output_path']+'/result_terafly',x_length=int(x[0]),y_length=int(x[1]),planes_num_0=int(x[2]),rescale_factor=int(pow(2,cur_res)/multiple),real_factors=pow(2,cur_res),mode=1)
                    out_terafly_part.run()
                if(self.jsoninfo['res_list'][str(cur_res)]['part_atitch']):
                    x = self.jsoninfo['res_list'][str(cur_res)]['res'].split("x")
                    out_terafly_part=output_terafly_c(data=self.jsoninfo,posjson_path=self.jsoninfo['output_path']+'/LR-z_y_x_p.json',input_folder=self.jsoninfo['input_path'],
                                                     result_folder=self.jsoninfo['output_path']+'/result_terafly',x_length=int(x[0]),y_length=int(x[1]),planes_num_0=int(x[2]),rescale_factor=int(pow(2,cur_res)/multiple),real_factors=pow(2,cur_res),mode=2)
                    out_terafly_part.run()
                    
            self.finishSignal.emit(int((cur_res+1)/6*100))
                
                    
                    
  
