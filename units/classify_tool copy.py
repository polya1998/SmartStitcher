from skimage import io
import os
import numpy
import pandas
from skimage import filters

def classify_compute(folder_path,file_list,indices_positve,indices_negative,row,col):
    cnt_p=len(indices_positve)
    cnt_n=len(indices_negative)
    pos_data = numpy.array([])
    pos_max_array = numpy.zeros(cnt_p)
    pos_mean_array = numpy.zeros(cnt_p)
    pos_std_array = numpy.zeros(cnt_p)
    index_p=0
    for pos in indices_positve:
        print(pos)
        file_name=file_list[pos[0]][pos[1]]+'_mip_z.tiff'
        file_path=folder_path+'/'+file_name
        im = (io.imread(file_path)).astype(float)
        if pos_data.size == 0:
            pos_data=im
            pos_max_array[0] = im.max()
            pos_mean_array[0] = im.mean()
            pos_std_array[0] = im.std()        
        else:
            pos_data=numpy.concatenate((pos_data, im), axis=0)
            pos_max_array[index_p] = im.max()
            pos_mean_array[index_p] = im.mean()
            pos_std_array[index_p] = im.std()
        
        index_p=index_p+1
    neg_data = numpy.array([])
    neg_max_array = numpy.zeros(cnt_n)
    neg_mean_array = numpy.zeros(cnt_n)
    neg_std_array = numpy.zeros(cnt_n)

    index_p = 0
    for pos in indices_negative:
        print(pos)
        file_name=file_list[pos[0]][pos[1]]+'_mip_z.tiff'
        file_path=folder_path+'/'+file_name
        #print(filename)
        im = (io.imread(file_path)).astype(float)
        if neg_data.size == 0:
            neg_data=im
            neg_max_array[0] = im.max()
            neg_mean_array[0] = im.mean()
            neg_std_array[0] = im.std()        
        else:
            neg_data=numpy.concatenate((neg_data, im), axis=0)
            neg_max_array[index_p] = im.max()
            neg_mean_array[index_p] = im.mean()
            neg_std_array[index_p] = im.std()
        
        index_p=index_p+1
    # calculate the threshold assuming two Gaussian distribution for background and foreground
    # same for Max,Mean and std
    thres_max = neg_max_array.max() + (pos_max_array.max()-neg_max_array.max())*neg_max_array.std()/(neg_max_array.std()+pos_max_array.std())

    thres_mean = neg_mean_array.mean() + (pos_mean_array.mean()-neg_mean_array.mean())*neg_mean_array.std()/(neg_mean_array.std()+pos_mean_array.std())

    thres_std = neg_std_array.max() + (pos_std_array.max()-neg_std_array.max())*neg_std_array.std()/(neg_std_array.std()+pos_std_array.std())
    thres_seg = filters.threshold_otsu(pos_data)
    # bright pixel number threshold as 0.2% of the volume voxels
    thres_birght_pixel_num = (im.size*0.002) # actually 0<this<2% all works
    # Start testing with new testing folder from a second brain
    result=numpy.zeros((int(row),int(col)))
    index=0
    for i in range (row) :
        for j in range (col):
            filename=file_list[i][j]+'_mip_z.tiff'
            file_path=folder_path+'/'+filename
            im = (io.imread(file_path)).astype(float)
            # if(numpy.mean(im)>thres_mean and (numpy.max(im))>thres_max and numpy.std(im)>thres_std and (im[im>thres_seg]).size>thres_birght_pixel_num):
            if(numpy.mean(im)>thres_mean):
                result[int(index/col)][index%col]=1
            else:
                result[int(index/col)][index%col]=-1
            index+=1
    return result
            
            
            