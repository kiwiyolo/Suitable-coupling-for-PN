# -*- coding: UTF-8 -*-
#
# Copyright (C) 2023 - 2023 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2023/11/6 15:46
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : data_Importance Sampling Scheme
# @IDE     : PyCharm
# -----------------------------------------------------------------
import numpy as np
from tqdm import tqdm
import netCDF4 as nc
import os
import json

qn_list, sample_list, sample_names = [], [], []     # 分别用于存放，样本的接受概率，样本数据，样本名称

def rain_sat_cal(x_nc,s):
    sat_x_nc = 1 - np.exp(-x_nc/s)
    return sat_x_nc


def acc_pro_cal(x_1d, samp_para):
    qmin = samp_para[0]
    m = samp_para[1]
    s = samp_para[2]
    C = len(x_1d)
    sum_sat = 0
    for c in range(C):
        sum_sat += rain_sat_cal(x_1d[c], s)
    q = qmin + m/C*sum_sat
    qn = min(1, q)
    return qn


def samplen_acc_pro(rainfall_map, samp_para, file_name):
    # 选择采样窗口的大小和偏移
    sample_size = (128, 128)
    offset = (32, 32)
    x_start = y_start = 0
    w = 0
    global qn_list, sample_list, sample_names

    # 创建tqdm对象
    total_iterations = ((rainfall_map.shape[1] - sample_size[0]) / offset[0] + 1) * (
            (rainfall_map.shape[2] - sample_size[1]) / offset[1] + 1)
    progress_bar = tqdm(total=total_iterations, desc='计算接收概率qn...', dynamic_ncols=True)

    while True:
        # 提取采样窗口
        sampled_window = rainfall_map[:, x_start:x_start + sample_size[0], y_start:y_start + sample_size[1]]
        sampled_1d = sampled_window.flatten(order='C')
        qn = acc_pro_cal(sampled_1d, samp_para)
        qn_list.append(qn)
        sample_list.append(sampled_window)
        sample_names.append(file_name + '_%s' % (w + 1))
        x_start += offset[0]
        if x_start > rainfall_map.shape[1] - sample_size[0]:
            x_start = 0
            y_start += offset[1]
        progress_bar.update(1)  # 更新进度条
        w += 1  # 更新窗口位数
        if y_start > rainfall_map.shape[2] - sample_size[1]:
            break

    # 关闭进度条
    progress_bar.close()


def importance_sampling(rainfall_map, samp_para, qn_avg):
    # 选择采样窗口的大小和偏移
    sample_size = (128, 128)
    offset = (32, 32)
    x_start = y_start = 0

    # 创建tqdm对象
    total_iterations = ((rainfall_map.shape[1] - sample_size[0]) / offset[0] + 1) * (
                (rainfall_map.shape[2] - sample_size[1]) / offset[1] + 1)
    progress_bar = tqdm(total=total_iterations, desc='根据均值采样...', dynamic_ncols=True)

    dataset_window = []
    while True:
        # 提取采样窗口
        sampled_window = rainfall_map[:, x_start:x_start + sample_size[0], y_start:y_start + sample_size[1]]
        sampled_1d = sampled_window.flatten(order='C')
        qn = acc_pro_cal(sampled_1d, samp_para)
        if qn >= qn_avg:
            dataset_window.append(sampled_window)
        x_start += offset[0]
        if x_start > rainfall_map.shape[1] - sample_size[0]:
            x_start = 0
            y_start += offset[1]
        progress_bar.update(1)  # 更新进度条
        if y_start > rainfall_map.shape[2] - sample_size[1]:
            break

    # 关闭进度条
    progress_bar.close()
    return dataset_window


def split_list(lst, n):
    """
    Divide the list lst into multiple sublists of length n
    :param lst: The list to divide
    :param n: The length of the sublist
    :return: A list of partitioned sublists
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def seq_windows(data, i, seq_len):
    """
    滑动窗口建立序列数据→多个输入
    :param data: 输入的待处理数据
    :param i: 第i个滑块
    :param seq_len: 滑块序列长度
    :return: 返回序列数据（特征+目标）window
    """
    window = data[i:i + seq_len]
    return window


def load_data(file_path, file_n):
    """
    加载数据文件路径，依次读取，转换成x，y
    :param num_data: 需要加载的数据量
    :return: 将返回值x, y赋给self.data, self.label
    """
    num_read = 0
    dataset = []
    for file_name in file_n:
        # if file_name.endswith("nc"):
        file = file_path + file_name
        data = nc.Dataset(file)  # open the dataset
        precip_data = data.variables['precipitationCal'][0].data
        precip_data[precip_data < 0] = 0

        dataset.append(precip_data)

        num_read += 1

    R = np.array(dataset)

    R[np.isnan(R)] = 0 # 缺失值处理：-9999.9为缺失值
    return R, file_name


def main():
    # 获取当前脚本文件所在的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 构建上一级文件夹的路径
    parent_path = os.path.dirname(current_path)
    # configs = json.load(open(parent_path + '/config_NcN.json', 'r', encoding='utf-8', errors='ignore'))  # 加载模拟配置json
    file_path = parent_path + '\kw_01_data/china_15-23_0.5s_train_IP/'
    samp_para = (2E-4, 0.1, 1)
    file_list = os.listdir(file_path)
    file_list = file_list[:36]
    file_group = split_list(file_list, 18)
    for file_n in file_group:
        sequence, file_name = load_data(file_path, file_n)
        samplen_acc_pro(sequence, samp_para, file_name)
    qn_avg = np.average(qn_list)
    print(f'扫描完成，接受度均值为：{qn_avg:2f}')
    for n in range(len(qn_list)):
        if qn_list[n] >= qn_avg:
            np.save(parent_path + '/kw_01_data/china_15-23_0.5s-IP_Imp-sampling/' + sample_names[n], sample_list[n])
            print(sample_names[n] + ':采样完成！')


if __name__ == '__main__':
    main()






