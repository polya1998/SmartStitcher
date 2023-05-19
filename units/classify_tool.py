from skimage import io
import os
import numpy
import pandas
from skimage import filters


import os
import numpy as np
import tifffile as tiff
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from scipy.fftpack import fft2
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import skimage.feature
from scipy.stats import kurtosis, skew, entropy
from skimage.filters import sobel
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import morphology
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.draw import line
import numpy as np
import heapq

# 广度优先搜索
def bfs(matrix, start, end):
    queue = Queue()
    queue.put((start, []))
    visited = set()
    while not queue.empty():
        (x, y), path = queue.get()
        if (x, y) == end:
            return path
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and (nx, ny) not in visited:
                queue.put(((nx, ny), path + [(nx, ny)]))
                visited.add((nx, ny))
    return []
def calculate_features(image_data) :
    # 确保图像为 32 位单通道 TIFF 图像
    # assert image.mode == "I"

    # image_data = np.array(image)

    # 计算亮度相关特征
    features=[]
    mean_brightness = float(np.mean(image_data))  # 平均亮度
    features.append(mean_brightness)
    max_brightness = float(np.max(image_data))  # 最大亮度
    features.append(max_brightness)
    min_brightness = float(np.min(image_data))  # 最小亮度
    features.append(min_brightness)
    brightness_std = float(np.std(image_data))  # 亮度标准差
    features.append(brightness_std)
    brightness_median = float(np.median(image_data))  # 亮度中位数
    features.append(brightness_median)
    

    local_entropy = float(shannon_entropy(image_data))  # 局部熵
    features.append(local_entropy)
    brightness_kurtosis = float(kurtosis(image_data.flatten()))  # 亮度峰度
    features.append(brightness_kurtosis)
    brightness_skewness = float(skew(image_data.flatten()))  # 亮度偏度
    features.append(brightness_skewness)


    # Sobel 梯度
    sobel_image = sobel(image_data)
    mean_gradient = float(np.mean(sobel_image))
    features.append(mean_gradient)
    std_gradient = float(np.std(sobel_image))
    features.append(std_gradient)

    # GLCM 特征
    scaled_image_data = ((image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 256).astype(np.uint8)
    levels = 2 ** 8
    glcm = skimage.feature.graycomatrix(scaled_image_data, [1], [0], levels=levels, symmetric=True, normed=True)
    contrast = float(skimage.feature.graycoprops(glcm, 'contrast')[0, 0])
    features.append(contrast)
    dissimilarity = float(skimage.feature.graycoprops(glcm, 'dissimilarity')[0, 0])
    features.append(dissimilarity)
    homogeneity = float(skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0])
    features.append(homogeneity)
    energy = float(skimage.feature.graycoprops(glcm, 'energy')[0, 0])
    features.append(energy)
    correlation = float(skimage.feature.graycoprops(glcm, 'correlation')[0, 0])
    features.append(correlation)

    return features

def load_images(folder,folder_list, label):
    features = []
    labels = []
    for file in folder_list:
        if file.endswith('.tiff'):
            img = tiff.imread(os.path.join(folder, file))
            features.append(calculate_features(img))
            labels.append(label)
    return features, labels

def predict_image(scaler,selector,svm,file):
    img = tiff.imread(file)
    features = calculate_features(img)
    scaled_features_matrix = scaler.transform([features])
    selected_features = selector.transform(scaled_features_matrix)
    return svm.predict(selected_features)[0]


def erode_matrix(matrix, window_size):

    # 找到所有岛屿
    islands = []
    visited = set()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1 and (i, j) not in visited:
                island = [(i, j)]
                stack = [(i, j)]
                visited.add((i, j))
                while stack:
                    x, y = stack.pop()
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and matrix[nx][ny] == 1 and (nx, ny) not in visited:
                            stack.append((nx, ny))
                            visited.add((nx, ny))
                            island.append((nx, ny))
                islands.append(island)

    # 找到面积最大的岛屿
    max_island = max(islands, key=len)

    # 将非最大岛屿全部变为0
    for island in islands:
        if island is not max_island:
            for x, y in island:
                matrix[x][y] = 0

    return matrix

def bresenham_line(matrix, point1, point2):
    rr, cc = line(point1[0], point1[1], point2[0], point2[1])
    matrix[rr, cc] = 1
    return matrix

# def connect_ones(matrix):
#     # 获取所有 1 的坐标
#     ones_coords = np.argwhere(matrix == 1)
    
#     # 计算所有 1 之间的距离矩阵
#     dist_matrix = distance_matrix(ones_coords, ones_coords)
    
#     # 生成最小生成树
#     mst = minimum_spanning_tree(dist_matrix).toarray()
    
#     # 连接所有的 1
#     for i in range(len(ones_coords)):
#         for j in range(i + 1, len(ones_coords)):
#             if mst[i, j] != 0:  # 如果两个点在最小生成树中相连
#                 matrix = bresenham_line(matrix, ones_coords[i], ones_coords[j])
                
#     return matrix

def connect_ones(matrix):
    # 找到所有的岛屿
    islands = []
    visited = set()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1 and (i, j) not in visited:
                island = [(i, j)]
                stack = [(i, j)]
                visited.add((i, j))
                while stack:
                    x, y = stack.pop()
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and matrix[nx][ny] == 1 and (nx, ny) not in visited:
                            stack.append((nx, ny))
                            visited.add((nx, ny))
                            island.append((nx, ny))
                islands.append(island)

    # 找到面积最大的岛屿
    max_island = max(islands, key=len)

    # 连接其他岛屿到最大岛屿
    for island in islands:
        if island is not max_island:
            start = island[0]
            end = max_island[0]
            path = bfs(matrix, start, end)
            for x, y in path:
                matrix[x][y] = 1

    return matrix

def classify_compute(folder_path,file_list,indices_positve,indices_negative,row,col):
    
    folder_all_list=os.listdir(folder_path)
    features_all, labels_pos = load_images(folder_path,folder_all_list, 1)
    scaler = MinMaxScaler()
    scaler.fit_transform(features_all)

    cnt_p=len(indices_positve)
    cnt_n=len(indices_negative)
    pos_list_train=[]
    for pos in indices_positve:
        print(pos)
        file_name=file_list[pos[0]][pos[1]]+'_mip_z.tiff'
        pos_list_train.append(file_name)
        
    neg_list_train=[]
    for pos in indices_negative:
        print(pos)
        file_name=file_list[pos[0]][pos[1]]+'_mip_z.tiff'
        neg_list_train.append(file_name)
        

        
        
    features_pos, labels_pos = load_images(folder_path,pos_list_train, 1)
    features_neg, labels_neg = load_images(folder_path,neg_list_train, 0)
    
    features_train = features_pos + features_neg
    labels_train = labels_pos + labels_neg
    
    
    scaled_features_matrix = scaler.transform(features_train)
    selector = SelectKBest(chi2, k=6)
    selected_features = selector.fit_transform(scaled_features_matrix, labels_train)
    selected_features_mask = selector.get_support()
    print("Selected features mask:", selected_features_mask)
    
    X_train=selected_features
    y_train=labels_train
    
    svm = SVC()
    svm.fit(X_train, y_train)
    result=numpy.zeros((int(row),int(col)))
    index=0
    for i in range (row) :
        for j in range (col):
            filename=file_list[i][j]+'_mip_z.tiff'
            file_path=folder_path+'/'+filename
            prediction = predict_image(scaler,selector,svm,file_path)
            # if(numpy.mean(im)>thres_mean and (numpy.max(im))>thres_max and numpy.std(im)>thres_std and (im[im>thres_seg]).size>thres_birght_pixel_num):
            if(prediction==1):
                result[int(index/col)][index%col]=1
            else:
                result[int(index/col)][index%col]=-1
            index+=1
    return result
            
            
            