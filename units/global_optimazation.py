import json
from sympy import *
import os
def G_O_Z(jason_file_path,save_path):
    json_env = jason_file_path
    with open(json_env, 'r')as fp:
        json_data = json.load(fp)
    eqs = json_data['result_z_yx_s']
    my_symbols = set()
    for i in eqs:
        my_symbols.add(i[0])
        my_symbols.add(i[1])
    my_symbols_str = ''
    for i in my_symbols:
        my_symbols_str = my_symbols_str + i + ' '
    my_symbols_str = my_symbols_str[0:-1]

    my_symbols = symbols(my_symbols_str)

    lin_eqs = []

    for i in my_symbols:
        tt = 0.0
        for j in eqs:
            if str(i) == j[0]:
                tt = tt + i - Symbol(j[1]) + j[2]
            if str(i) == j[1]:
                tt = tt + i - Symbol(j[0]) - j[2]
        lin_eqs.append(Eq(tt, 0))
    lin_eqs.append(Eq(my_symbols[0], 0))
    print(lin_eqs)
    r = linsolve(lin_eqs, my_symbols)
    d = {}
    for i in range(len(my_symbols)):
        for j in r:
            print(j[i])
            d[str(my_symbols[i])] = round(float(j[i]), 2)
    with open(save_path,'w') as f:
        json.dump(d,f,indent=4,ensure_ascii=False)
    print(r)
def G_O_Y(jason_file_path,save_path):
    json_env = jason_file_path

    with open(json_env, 'r')as fp:
        json_data = json.load(fp)
    eqs = json_data["result_y_x_s"]
    my_symbols = set()
    for i in eqs:
        my_symbols.add(i[0])
        my_symbols.add(i[1])
    my_symbols_str = ''
    for i in my_symbols:
        my_symbols_str = my_symbols_str + i + ' '
    my_symbols_str = my_symbols_str[0:-1]

    my_symbols = symbols(my_symbols_str)

    lin_eqs = []

    for i in my_symbols:
        tt = 0.0
        for j in eqs:
            if str(i) == j[0]:
                tt = tt + i - Symbol(j[1]) + j[2]
            if str(i) == j[1]:
                tt = tt + i - Symbol(j[0]) - j[2]
        lin_eqs.append(Eq(tt, 0))
    lin_eqs.append(Eq(my_symbols[0], 0))
    print(lin_eqs)
    r = linsolve(lin_eqs, my_symbols)
    d = {}
    for i in range(len(my_symbols)):
        for j in r:
            print(j[i])
            d[str(my_symbols[i])] = round(float(j[i]), 2)
    with open(save_path,'w') as f:
        json.dump(d,f,indent=4,ensure_ascii=False)
    print(r)
    
def G_O_X(jason_file_path,save_path):
    json_env = jason_file_path

    with open(json_env, 'r')as fp:
        json_data = json.load(fp)
    eqs = json_data["result_y_x_s"]
    my_symbols = set()
    for i in eqs:
        my_symbols.add(i[0])
        my_symbols.add(i[1])
    my_symbols_str = ''
    for i in my_symbols:
        my_symbols_str = my_symbols_str + i + ' '
    my_symbols_str = my_symbols_str[0:-1]

    my_symbols = symbols(my_symbols_str)

    lin_eqs = []

    for i in my_symbols:
        tt = 0.0
        for j in eqs:
            if str(i) == j[0]:
                tt = tt + i - Symbol(j[1]) + j[3]
            if str(i) == j[1]:
                tt = tt + i - Symbol(j[0]) - j[3]
        lin_eqs.append(Eq(tt, 0))
    lin_eqs.append(Eq(my_symbols[0], 0))
    print(lin_eqs)
    r = linsolve(lin_eqs, my_symbols)
    d = {}
    for i in range(len(my_symbols)):
        for j in r:
            d[str(my_symbols[i])] = round(float(j[i]), 2)
    with open(save_path,'w') as f:
        json.dump(d,f,indent=4,ensure_ascii=False)
    print(r) 



def apply_highres_offsets_to_lowres_G_O_Z(highres_json_path, lowres_json_path, save_path, rescale_factor):
    if(os.path.exists(save_path)):
        return
    # 加载高分辨率优化结果 JSON
    with open(highres_json_path, 'r') as fp:
        highres_offsets = json.load(fp)

    # 加载低分辨率 JSON
    with open(lowres_json_path, 'r') as fp:
        lowres_json_data = json.load(fp)

    lowres_eqs = lowres_json_data['result_z_yx_s']
    
    # 创建高分辨率符号集
    highres_symbols_set = set()
    for sym in highres_offsets:
        highres_symbols_set.add(sym)
        
    highres_symbols = symbols(' '.join(highres_symbols_set))
    # 创建低分辨率符号集
    lowres_symbols_set = set()
    for i in lowres_eqs:
        lowres_symbols_set.add(i[0])
        lowres_symbols_set.add(i[1])

    lowres_symbols_str = ' '.join(lowres_symbols_set)
    lowres_symbols = symbols(lowres_symbols_str)

    # 为低分辨率图像创建线性方程
    lowres_lin_eqs = []
    for i in lowres_symbols:
        if i in highres_symbols:
            continue
        tt = 0.0
        for j in lowres_eqs:
            if str(i) == j[0]:
                tt = tt + i - Symbol(j[1]) + j[2]
            if str(i) == j[1]:
                tt = tt + i - Symbol(j[0]) - j[2]
        lowres_lin_eqs.append(Eq(tt, 0))

    # # 将第一个低分辨率符号的偏移量设置为0
    # lowres_lin_eqs.append(Eq(lowres_symbols[0], 0))

    # 将高分辨率的优化结果应用到重叠区域
    for sym in highres_offsets:
        if sym in [str(s) for s in lowres_symbols]:
            lowres_lin_eqs.append(Eq(symbols(sym) - highres_offsets[sym] / rescale_factor, 0))

    # 求解线性方程组
    lowres_solution = linsolve(lowres_lin_eqs, lowres_symbols)
    
    # 将结果保存为字典
    lowres_solution_dict = {}
    for i, sym in enumerate(lowres_symbols):
        for sol in lowres_solution:
            lowres_solution_dict[str(sym)] = round(float(sol[i]), 2)

    # 将结果保存为 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(lowres_solution_dict, f, indent=4, ensure_ascii=False)

    return lowres_solution_dict

def apply_highres_offsets_to_lowres_G_O_Y(highres_json_path, lowres_json_path, save_path, rescale_factor):
    # 加载高分辨率优化结果 JSON
    if(os.path.exists(save_path)):
        return
    with open(highres_json_path, 'r') as fp:
        highres_offsets = json.load(fp)

    # 加载低分辨率 JSON
    with open(lowres_json_path, 'r') as fp:
        lowres_json_data = json.load(fp)

    lowres_eqs = lowres_json_data['result_y_x_s']
    
    # 创建高分辨率符号集
    highres_symbols_set = set()
    for sym in highres_offsets:
        highres_symbols_set.add(sym)
        
    highres_symbols = symbols(' '.join(highres_symbols_set))
    # 创建低分辨率符号集
    lowres_symbols_set = set()
    for i in lowres_eqs:
        lowres_symbols_set.add(i[0])
        lowres_symbols_set.add(i[1])

    lowres_symbols_str = ' '.join(lowres_symbols_set)
    lowres_symbols = symbols(lowres_symbols_str)

    # 为低分辨率图像创建线性方程
    lowres_lin_eqs = []
    for i in lowres_symbols:
        if i in highres_symbols:
            continue
        tt = 0.0
        for j in lowres_eqs:
            if str(i) == j[0]:
                tt = tt + i - Symbol(j[1]) + j[2]
            if str(i) == j[1]:
                tt = tt + i - Symbol(j[0]) - j[2]
        lowres_lin_eqs.append(Eq(tt, 0))

    # # 将第一个低分辨率符号的偏移量设置为0
    # lowres_lin_eqs.append(Eq(lowres_symbols[0], 0))

    # 将高分辨率的优化结果应用到重叠区域
    for sym in highres_offsets:
        if sym in [str(s) for s in lowres_symbols]:
            lowres_lin_eqs.append(Eq(symbols(sym) - highres_offsets[sym] / rescale_factor, 0))

    # 求解线性方程组
    lowres_solution = linsolve(lowres_lin_eqs, lowres_symbols)

    # 将结果保存为字典
    lowres_solution_dict = {}
    for i, sym in enumerate(lowres_symbols):
        for sol in lowres_solution:
            lowres_solution_dict[str(sym)] = round(float(sol[i]), 2)

    # 将结果保存为 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(lowres_solution_dict, f, indent=4, ensure_ascii=False)

    return lowres_solution_dict

def apply_highres_offsets_to_lowres_G_O_X(highres_json_path, lowres_json_path, save_path, rescale_factor):
    if(os.path.exists(save_path)):
        return
    # 加载高分辨率优化结果 JSON
    with open(highres_json_path, 'r') as fp:
        highres_offsets = json.load(fp)

    # 加载低分辨率 JSON
    with open(lowres_json_path, 'r') as fp:
        lowres_json_data = json.load(fp)

    lowres_eqs = lowres_json_data['result_y_x_s']
    # 创建高分辨率符号集
    highres_symbols_set = set()
    for sym in highres_offsets:
        highres_symbols_set.add(sym)
        
    highres_symbols = symbols(' '.join(highres_symbols_set))
    # 创建低分辨率符号集
    lowres_symbols_set = set()
    for i in lowres_eqs:
        lowres_symbols_set.add(i[0])
        lowres_symbols_set.add(i[1])

    lowres_symbols_str = ' '.join(lowres_symbols_set)
    lowres_symbols = symbols(lowres_symbols_str)

    # 为低分辨率图像创建线性方程
    lowres_lin_eqs = []
    for i in lowres_symbols:
        if i in highres_symbols:
            continue
        tt = 0.0
        for j in lowres_eqs:
            if str(i) == j[0]:
                tt = tt + i - Symbol(j[1]) + j[3]
            if str(i) == j[1]:
                tt = tt + i - Symbol(j[0]) - j[3]
        lowres_lin_eqs.append(Eq(tt, 0))

    # # 将第一个低分辨率符号的偏移量设置为0
    # lowres_lin_eqs.append(Eq(lowres_symbols[0], 0))

    # 将高分辨率的优化结果应用到重叠区域
    for sym in highres_offsets:
        if sym in [str(s) for s in lowres_symbols]:
            lowres_lin_eqs.append(Eq(symbols(sym) - highres_offsets[sym] / rescale_factor, 0))

    # 求解线性方程组
    lowres_solution = linsolve(lowres_lin_eqs, lowres_symbols)

    # 将结果保存为字典
    lowres_solution_dict = {}
    for i, sym in enumerate(lowres_symbols):
        for sol in lowres_solution:
            lowres_solution_dict[str(sym)] = round(float(sol[i]), 2)

    # 将结果保存为 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(lowres_solution_dict, f, indent=4, ensure_ascii=False)

    return lowres_solution_dict