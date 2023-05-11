import json
def zz(json_path,save_path):
    json_env = json_path
    with open(json_env, 'r')as fp:
        json_data = json.load(fp)
    d = {}
    my_min = 0
    for i in json_data.keys():
        if json_data[i]<my_min:
            my_min=json_data[i]
    for i in json_data.keys():
        json_data[i] = json_data[i] - my_min
    for i in json_data.keys():
        d[i] = round(json_data[i])
    with open(save_path,'w') as f:
        json.dump(d,f,indent=4,ensure_ascii=False)

def yy(json_path,save_path):
    json_env = json_path
    with open(json_env, 'r')as fp:
        json_data = json.load(fp)
    d = {}
    my_min = 0
    for i in json_data.keys():
        if json_data[i]<my_min:
            my_min=json_data[i]
    for i in json_data.keys():
        json_data[i] = json_data[i] - my_min
    for i in json_data.keys():
        d[i] = round(json_data[i])
    with open(save_path,'w') as f:
        json.dump(d,f,indent=4,ensure_ascii=False)

def xx(json_path,save_path):
    json_env = json_path
    with open(json_env, 'r')as fp:
        json_data = json.load(fp)
    d = {}
    my_min = 0
    for i in json_data.keys():
        if json_data[i]<my_min:
            my_min=json_data[i]
    for i in json_data.keys():
        json_data[i] = json_data[i] - my_min
    for i in json_data.keys():
        d[i] = round(json_data[i])
    with open(save_path,'w') as f:
        json.dump(d,f,indent=4,ensure_ascii=False)

def write_z_y_x(src_path,save_path,x_length,y_length,isHR):

    if(isHR):
        json_env = src_path+'/HR-Z-shift_manual_g_o_position_z.json'

        with open(json_env, 'r')as fp:
            json_data_z = json.load(fp)
        json_env = src_path+'/HR-Y-shift_manual_g_o_position_y.json'

        with open(json_env, 'r')as fp:
            json_data_y = json.load(fp)

        json_env = src_path+'/HR-X-shift_manual_g_o_position_x.json'

        with open(json_env, 'r')as fp:
            json_data_x = json.load(fp)
    if(not isHR):
        json_env = src_path+'/LR-Z-shift_manual_g_o_position_z.json'

        with open(json_env, 'r')as fp:
            json_data_z = json.load(fp)
        json_env = src_path+'/LR-Y-shift_manual_g_o_position_y.json'

        with open(json_env, 'r')as fp:
            json_data_y = json.load(fp)

        json_env = src_path+'/LR-X-shift_manual_g_o_position_x.json'

        with open(json_env, 'r')as fp:
            json_data_x = json.load(fp)

    tiles=[]
    for i in json_data_z.keys():
        tiles.append(i)
    tiles.sort()
    d=[]
    for i in tiles:
        t_d={}
        t_d['Tile'] = i
        t_d['z'] = json_data_z[i]
        t_d['y'] = json_data_y[i]#注意命名不统一
        t_d['x'] = json_data_x[i]#注意命名不统一
        t_d['x_length'] = x_length
        t_d['y_length'] = y_length
        d.append(t_d)

    dd={}
    dd['z_y_x_p'] = d
    with open(save_path, 'w') as f:
        json.dump(dd, f, indent=4, ensure_ascii=False)