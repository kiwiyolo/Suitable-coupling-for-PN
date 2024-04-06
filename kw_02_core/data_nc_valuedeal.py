import torch
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import os
import shutil



def mean_max_min(folder_path, sequence_file_path):
    # 获取文件夹中所有nc4文件的路径
    nc4_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".nc4")]
    sequence_length = 18
    data_num = 0
    mean_list, max_list, min_list = [], [], []
    # 遍历每个nc4文件并计算均值
    sequence_file = 'data_sequence_inf_selected.txt'
    f = open(sequence_file_path + sequence_file,'a')
    f.write("sequence_end_File,sequence_mean,sequence_max,sequence_min")
    f.write('\n')
    for nc4_file in nc4_files:
        # 打开nc4文件
        dataset = Dataset(nc4_file, "r")
        data_num += 1
        # 读取数据（假设你要计算的数据在名为"variable_name"的变量中）
        variable_name = "precipitationCal"  # 请替换为实际的变量名
        data = dataset.variables[variable_name][:]

        # 将数据转换为PyTorch张量
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 计算均值、最大值、最小值
        mean_value = torch.mean(data_tensor).item()
        max_value = torch.max(data_tensor).item()
        min_value = torch.min(data_tensor).item()

        # 输出文件名和均值
        print(f"File: {nc4_file.split('/')[-1]}, Mean: {mean_value}, Max: {max_value}, Min: {min_value}")
        mean_list.append(mean_value)
        max_list.append(max_value)
        min_list.append(min_value)

        if data_num == sequence_length:
            sequence_mean = sum(mean_list)/len(mean_list)
            sequence_max = max(max_list)
            sequence_min = min(min_list)
            sequence_inf = f"{nc4_file.split('/')[-1]},{sequence_mean},{sequence_max},{sequence_min}"
            print(sequence_inf)
            f.write(sequence_inf)
            f.write('\n')
            data_num = 0
            mean_list = []
            max_list = []
        # 关闭nc4文件
        dataset.close()
    f.close()


def meanmax_quartile(file_csv, column_index):
    # 获取当前脚本文件所在的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 构建上一级文件夹的路径
    parent_path = os.path.dirname(current_path)
    # 构建CSV的相对路径
    file_path = os.path.join(parent_path, 'kw_01_data/data_20200101_20230531/try1_china_TVT',
                             file_csv)

    # 使用pandas读取CSV文件
    data = pd.read_csv(file_path)
    # 提取需要计算的列数据
    column_data = data.iloc[:, column_index]
    # 计算四分之一分位数
    first_quartile = np.percentile(column_data, 25)
    return first_quartile


def data_valid_selected(file_csv, data_path, target_file_path):
    # 获取当前脚本文件所在的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 构建上一级文件夹的路径
    parent_path = os.path.dirname(current_path)
    # 构建CSV的相对路径
    file_path = os.path.join(parent_path, 'kw_01_data/data_20200101_20230531/try1_china_TVT',
                             file_csv)
    # 使用pandas读取CSV文件
    data = pd.read_csv(file_path)
    # 提取需要计算的列数据
    column_data = data.iloc[:, 0]
    target_sequences = column_data.tolist()

 # 获取文件夹中所有nc4文件的路径
    nc4_files = [f for f in os.listdir(data_path) if f.endswith(".nc4")]
    sequence_length = 18
    data_num = 0
    selected_list = []
    target_list = []
    # target_files记录所筛选的有效值
    target_files = 'data_valid_selected.txt'

    for nc4_file in nc4_files:
        selected_list.append(nc4_file)
        data_num += 1

        if data_num == sequence_length:
            if nc4_file in target_sequences:
                target_list.extend(selected_list)
            else:
                print(f"invalid_files: {nc4_file}")
            data_num = 0
            selected_list = []
    f = open(target_file_path + target_files, 'a')
    f.write("target_files")
    f.write('\n')

    for target in target_list:
        f.write(target)
        f.write('\n')

    f.close()

def copy_to_new(source_folder,new_folder_path,target_files):

    # 原文件夹路径
    # 新文件夹路径
    # 文件名列表

    # 获取当前脚本文件所在的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 构建上一级文件夹的路径
    parent_path = os.path.dirname(current_path)
    file_path = os.path.join(parent_path, 'kw_01_data/data_20200101_20230531/try1_china_TVT',
                             target_files)

    # 使用pandas读取CSV文件
    data = pd.read_csv(file_path)
    # 提取需要计算的列数据
    column_data = data.iloc[:, 0]
    file_list = column_data.tolist()

    for filename in file_list:
        source_file_path = os.path.join(source_folder, filename)
        new_file_path = os.path.join(new_folder_path, filename)

        # 检查源文件是否存在
        if os.path.exists(source_file_path):
            # 复制粘贴文件
            shutil.copy(source_file_path, new_file_path)
            print(f"文件 {filename} 复制成功")
        else:
            print(f"文件 {filename} 不存在")

    print("复制粘贴完成")


def copy(source_folder, new_folder_path):
    for filename in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, filename)


        # 检查源文件是否存在
        if os.path.exists(source_file_path):
            newfilename = filename.replace(" ", "")
            new_file_path = os.path.join(new_folder_path, newfilename)
            # 复制粘贴文件
            shutil.copy(source_file_path, new_file_path)
            print(f"文件 {filename} 复制成功")
        else:
            print(f"文件 {filename} 不存在")

    print("复制粘贴完成")


if __name__ == '__main__':
    # 获取当前脚本文件所在的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 构建上一级文件夹的路径
    parent_path = os.path.dirname(current_path)

    # 1、计算出每个sequence的的均值、最大值、最小值并保存
    # 设置文件夹路径
    folder_path = parent_path + "/kw_01_data/data_20200101_20230531/try1_china_TVT/clip_china_precip_rect/"
    sequence_file_path = parent_path + '/kw_01_data/data_20200101_20230531/try1_china_TVT/'
    # mean_max_min(folder_path, sequence_file_path)

    # 2、先删去mean<0的sequences！！！！，根据sequence均值和最大值求取四分之一分位数
    # column_index = 2  # 你想要计算四分之一分位数的列索引
    # file_csv = 'data_sequence_inf_selected.csv'
    # first_quartile = meanmax_quartile(file_csv, column_index)
    # print("25分位数:", first_quartile)

    # 3、提取经过筛选后有效的文件文件名in：csv, out：txt
    # data_valid_selected(file_csv, folder_path, sequence_file_path)

    # 4、根据提取的文件名复制文件至新路径
    target_files = 'data_valid_selected.csv'
    new_folder_path = sequence_file_path + 'clip_china_precip_rect_0.25selected/'
    copy_to_new(folder_path, new_folder_path, target_files)
