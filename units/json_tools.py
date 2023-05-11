import json
import os


def readjson(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
            print(e)


def find_same_name(path):
        # 获取指定路径下的所有文件名
    file_list = os.listdir(path)
    # 将第一个文件名作为基准
    base_name = file_list[0]
    # 遍历文件名列表，找到相同部分和不同部分
    for file_name in file_list:
        i = 0
        while i < len(base_name) and i < len(file_name) and base_name[i] == file_name[i]:
            i += 1
        base_name = base_name[:i]
    # 将不同部分用*代替，得到命名规则
    name_format = base_name + '*'
    return name_format
      
def generate_json_file(file_path, data):
    """
    将数据写入到本地JSON文件中
    :param file_path: 本地JSON文件的文件路径
    :param data: 要写入JSON文件的数据
    :return: 无返回值
    """
    print(file_path)
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f,indent=4, ensure_ascii=False)
        return 1
    except Exception as e:
        print(e,"无法完成覆写")
        return -1

# data={
#                 "input_path": '',
#                 "output_path": '',
#                 "wildname": '',
#                 "x_length":0,
#                 "y_length":0,
#                 "z_length":0,
#                 "vixel_size":1.,
                
#         }
# file_path = 'test.json'
# generate_json_file(file_path, data)
# data=readjson(file_path)
# print(data)
