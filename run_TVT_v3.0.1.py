# ! /public/home/luoqun/p2_gloprec_DL/myenv/bin/python
# -*- coding: UTF-8 -*-
#
# Copyright (C) 2023 - 2023 QunLuo, Inc. All Rights Reserved
#
# @Time    : 2023/9/22 9:36
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : model_net_norm
# @IDE     : PyCharm
# -----------------------------------------------------------------
import json
import time as t
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from kw_02_core.data_processor import Data
from kw_02_core.models_class import CNcN, RNcN
from kw_02_core.model_ConvPLSTM import UNET_Like_3D, NET_ConvLSTM, UNET_mine
from kw_02_core.data_processor import Data_GPM
from kw_02_core.utils import EarlyStopping
from kw_02_core.plot_multi_prec_eval import  plot_eva_tiff
from pysteps.visualization import plot_precip_field
from pysteps import verification
import math
import re
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
import pandas as pd
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from tqdm import tqdm
import sys
import rasterio
from rasterio.transform import from_origin
import rioxarray


os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
torch.autograd.set_detect_anomaly(True)


def split_list(lst, n):
    """
    将列表 lst 划分为长度为 n 的多个子列表
    :param lst: 要划分的列表
    :param n: 子列表的长度
    :return: 划分后的子列表组成的列表
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def get_region(eva_point, regions):
    # Check which region the point falls into
    for region, geometry in regions.items():
        if eva_point.within(geometry):
            return region
    return None


def input_with_default(prompt, default_value):
    response = input(f"{prompt} (默认为 {default_value}): ")
    return response if response else default_value


def create_model(model_name, model_PSU, device):
    if model_name == "CNcN":
        # 创建模型实例
        model = CNcN((3, 1, 9, 640, 512), 16, 3, 'relu', 0.1, 3, pres=model_PSU).to(device)


    elif model_name == "RNcN":
        # 定义模型参数
        convlstm_input_channels = 1
        convlstm_hidden_channels = [16, 32, 64, 64, 32, 16]
        convlstm_kernel_size = 3
        conv3d_input_channels = convlstm_hidden_channels[0]
        conv3d_output_channels = [16, 32, 64, 64, 32, 16, 1]
        conv3d_kernel_size = 3

        # 创建模型实例
        model = RNcN(convlstm_input_channels, convlstm_hidden_channels, convlstm_kernel_size,
                     conv3d_input_channels, conv3d_output_channels, conv3d_kernel_size, pres=model_PSU).to(device)

    elif model_name == "UNET_Like_3D":
        # 创建模型实例
        model = UNET_Like_3D((3, 1, 9, 640, 512), 48, 3, 'relu', 0.1, 3, pres=model_PSU).to(device)

    elif model_name == "UNET_mine":
        # 创建模型实例
        model = UNET_mine((3, 1, 9, 640, 512), 24, 3, 'relu', 0.1, 3, pres=model_PSU).to(device)

    elif model_name == "NET_ConvLSTM":
        # 定义模型参数
        convlstm_input_channels = 1
        convlstm_hidden_channels = [16, 32, 16]
        convlstm_kernel_size = 3
        conv3d_input_channels = convlstm_hidden_channels[0]
        conv3d_output_channels = [32, 16, 2, 1]
        conv3d_kernel_size = 3

        # 创建模型实例
        model = NET_ConvLSTM(convlstm_input_channels, convlstm_hidden_channels, convlstm_kernel_size,
                             conv3d_input_channels, conv3d_output_channels, conv3d_kernel_size, pres=model_PSU).to(
            device)

    else:
        raise ValueError(f"定义的模型库中不存在该模型: {model_name}.")
    return model


def adjust_learning_rate(configs, epoch, optimizer, lr, paras):
    file_list = os.listdir(configs["data"]["file_path"])
    Estart, Te, tt, multFactor, t0 = paras
    # 调整学习率
    if (epoch + 1 >= Estart):  # time to start adjust learning tate
        dt = 2.0 * math.pi / float(2.0 * Te)
        dt_batch = ((float(len(file_list)) / configs['data']['sequence_length']) / float(
            configs["training"]["batch_size"]))
        tt = tt + float(dt) / dt_batch
        print("tt: ", tt)

        if tt >= math.pi:
            tt = tt - math.pi
        curT = t0 + tt
        lr = lr * (1.0 + math.sin(curT)) / 2.0  # lr_min = 0, lr_max = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 打印损失self.lr
        print('self.lr', lr)


def predicted_results(model, x_test):
    pred_total = []
    for p in range(len(x_test)):
        with torch.no_grad():
            pred_point = (model(x_test[p][np.newaxis, :, :, :, :])).cpu()  # 将运算结果从cuda调到cpu中
            pred_total.append(pred_point)
            del pred_point
    pred_total = torch.cat(pred_total, dim=0).cpu().numpy()  # 将张量从计算图中分离出来并转化为numpy array
    pred_total[pred_total < 0] = 0
    return pred_total


def calculate_r_squared(pred, obs):
    # Mask NaN values in both pred and obs
    mask = ~np.isnan(pred) & ~np.isnan(obs)

    # Calculate sums with NaN values ignored
    ss_res = np.sum((obs[mask] - pred[mask]) ** 2)
    ss_tot = np.sum((obs[mask] - np.mean(obs[mask])) ** 2)

    # Check if ss_tot is zero to avoid division by zero
    if ss_tot == 0:
        return np.nan
    else:
        return 1 - ss_res / ss_tot


def calculate_rbias(pred, obs):
    # Create masks to exclude NaN values
    pred_mask = ~np.isnan(pred)
    obs_mask = ~np.isnan(obs)

    # Calculate rbias only for non-NaN values
    rbias = (np.mean(pred[pred_mask]) - np.mean(obs[obs_mask])) / np.mean(obs[obs_mask])
    return rbias


def calculate_cc(pred, obs):
    # Create masks to exclude NaN values
    pred_mask = ~np.isnan(pred)
    obs_mask = ~np.isnan(obs)

    # Flatten arrays and calculate correlation coefficient only for non-NaN values
    cc = np.corrcoef(pred[pred_mask].ravel(), obs[obs_mask].ravel())[0, 1]
    return cc

# 注意：这里ACC是一个伪造的指标，实际上连续数据不适合计算ACC
def calculate_acc(pred, obs, threshold= 0.1):
    pred_binary = (pred > threshold).astype(int)
    obs_binary = (obs > threshold).astype(int)
    return np.mean(pred_binary == obs_binary)


def calculate_sal(pred, obs):
    sal = verification.get_method("SAL")
    sal_s, sal_a, sal_l, eva_sal = [], [], [], []
    for i in range(pred.shape[0]):
        single_sal = sal(pred[i], obs[i])
        sal_s.append(single_sal[0])
        sal_a.append(single_sal[1])
        sal_l.append(single_sal[2])
    eva_sal.append(sum(sal_s) / len(sal_s))
    eva_sal.append(sum(sal_a) / len(sal_a))
    eva_sal.append(sum(sal_l) / len(sal_l))
    return eva_sal


def calculate_metrics_all(pred_total, y_test, configs, scheme, thr):
    # 结果评价（4大类方法）
    # 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    # 配置参数
    thr = float(thr)
    shp_root = r".\kw_01_data\china_boundary/"
    shp_list = ['bou1_4p.shp']
    pred_total = pred_total[:, 0]
    y_test = y_test[:, 0]

    n_leadtimes = configs['data']['n_leadtimes']
    leadt = [j for j in range(n_leadtimes)]
    sample = [i for i in range(len(y_test))]
    # 使用字典推导式创建字典
    regions = {str(shp_name).split('.shp')[0]: shp_root + shp_name for shp_name in shp_list}
    outcome_path = configs["eval"]["outcome_path"] + 'global_metrics_all/'
    n_leadtimes = configs['data']['n_leadtimes']
    # 降水数据shape→构建经纬度坐标
    shape = (512, 640)

    # 创建对应的经纬度坐标
    latitudes = np.linspace(2.8, 54, shape[0])
    longitudes = np.linspace(72, 136, shape[1])


    # 创建 xarray 数据集，附加经纬度坐标信息
    pred_xr = xr.DataArray(pred_total, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})
    ture_xr = xr.DataArray(y_test, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})
    pred_xr.rio.set_crs("EPSG:4326", inplace=True)
    ture_xr.rio.set_crs("EPSG:4326", inplace=True)
    # 测试集评价结果的均值
    ## 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    pod = verification.get_method("POD")
    csi = verification.get_method("CSI")
    far = verification.get_method("FAR")
    hss = verification.get_method("HSS")
    mse = verification.get_method("MSE")
    sal = verification.get_method("SAL")

    # 附加地理信息


    for key, value in regions.items():
        shp = gpd.read_file(value).set_crs(epsg=4326)

        # 利用 shapefile 进行裁剪
        clipped_pred = pred_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)
        clipped_ture = ture_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)

        for j in range(n_leadtimes):
            file = outcome_path + f"eval_all_{scheme}_{30 * (j + 1)}.csv"
            f = open(file, 'a')
            f.write("POD,CSI,FAR,HSS,MSE,SAL(S),SAL(A),SAL(L),R-squared,RBIAS,CC,ACC")
            f.write("\n")

            for i in tqdm(range(len(y_test)), desc=f'calculate_metrics_all[{(j+1)*30}min]'):
                eva_pred, eva_obs = clipped_pred.values[i, j, :, :], clipped_ture.values[i, j, :, :]
                if np.nanmax(eva_obs) > 0.1:
                    eva_pod = float(pod(eva_pred, eva_obs, thr=thr)['POD'])
                    eva_csi = float(csi(eva_pred, eva_obs, thr=thr)['CSI'])
                    eva_far = float(far(eva_pred, eva_obs, thr=thr)['FAR'])
                    eva_hss = float(hss(eva_pred, eva_obs, thr=thr)['HSS'])

                    eva_mse = float(mse(eva_pred, eva_obs)['MSE'])
                    try:
                        eva_sal = sal(eva_pred, eva_obs)  # NaNs are ignored
                    except ZeroDivisionError:
                        print("=" * 70)
                        print(f"scheme:{scheme}, region:{key}, shp:{value}, 分母为0, 无法计算SAL得分")
                        print("=" * 70)
                        eva_sal = [np.nan, np.nan, np.nan]

                    # 使用上述函数进行计算
                    r_squared = calculate_r_squared(eva_pred, eva_obs)
                    rbias = calculate_rbias(eva_pred, eva_obs)
                    cc = calculate_cc(eva_pred, eva_obs)
                    acc = calculate_acc(eva_pred, eva_obs, thr)  # 这里的threshold可能需要调整
                    f.write(
                        f"{eva_pod}, {eva_csi}, {eva_far}, {eva_hss}, {eva_mse}, {eva_sal[0]}, {eva_sal[1]}, {eva_sal[2]}, {r_squared}, {rbias}, {cc}, {acc}")
                    f.write("\n")


def calculate_metrics_mean(pred_total, y_test, configs, scheme, thr):
    shp_root = r".\kw_01_data\china_boundary/"
    shp_list = ['bou1_4p.shp']

    # 使用字典推导式创建字典
    regions = {str(shp_name).split('.shp')[0]: shp_root + shp_name for shp_name in shp_list}
    # 结果评价（4大类方法）
    # 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    # 配置参数
    outcome_path = configs["eval"]["outcome_path"]

    n_leadtimes = configs['data']['n_leadtimes']
    leadt = [j for j in range(n_leadtimes)]
    sample = [i for i in range(len(y_test))]

    # 测试集评价结果的均值
    ## 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    pod = verification.get_method("POD")
    csi = verification.get_method("CSI")
    far = verification.get_method("FAR")
    hss = verification.get_method("HSS")
    mse = verification.get_method("MSE")
    # 调整数据结构，构建对应维度

    pred_total = pred_total[:, 0]
    y_test = y_test[:, 0]

    # 降水数据shape→构建经纬度坐标
    shape = (512, 640)

    # 创建对应的经纬度坐标
    latitudes = np.linspace(2.8, 54, shape[0])
    longitudes = np.linspace(72, 136, shape[1])

    # 创建 xarray 数据集，附加经纬度坐标信息
    pred_xr = xr.DataArray(pred_total, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})
    ture_xr = xr.DataArray(y_test, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})
    pred_xr.rio.set_crs("EPSG:4326", inplace=True)
    ture_xr.rio.set_crs("EPSG:4326", inplace=True)

    # # 保存为 NetCDF 文件（可选）
    # precip_xr.to_netcdf('precip_data.nc')

    # 遍历分区，依次裁剪计算指标
    for key, value in regions.items():
        shp = gpd.read_file(value).set_crs(epsg=4326)
        # shp = gpd.read_file(value).set_crs(epsg=4326)

        # 利用 shapefile 进行裁剪
        clipped_pred = pred_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)
        clipped_ture = ture_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)


        for j in range(n_leadtimes):
            file = outcome_path + f"eval_{scheme}_{30 * (j + 1)}.csv"
            f = open(file, 'a')
            f.write("Region,ACC,CSI,FAR,HSS,POD,CC,MSE,percent_p(%)")
            f.write("\n")

            # Initialize variables to store cumulative values and counts for each metric
            # Determine the region for the current prediction
            region = key
            # Create dictionaries to store lists for each metric
            num_p = 0
            metrics_dict = {
                'acc': [],
                'eva_csi': [],
                'eva_far': [],
                'eva_hss': [],
                'eva_pod': [],
                'cc': [],
                'eva_mse': [],
            }

            for i in tqdm(range(len(y_test)), desc=f'评价global_mean:[R]{key}_[T]{30 * (j + 1)}min中'):
                eva_pred, eva_obs = clipped_pred.values[i, j, :, :], clipped_ture.values[i, j, :, :]


                if region is not None and np.nanmax(eva_obs) > 0.1:
                    num_p += 1
                    # 各个指标在进行逻辑运算时，会将结果为nan的排除统计
                    eva_pod = float(pod(eva_pred, eva_obs, thr=thr)['POD'])
                    eva_csi = float(csi(eva_pred, eva_obs, thr=thr)['CSI'])
                    eva_far = float(far(eva_pred, eva_obs, thr=thr)['FAR'])
                    eva_hss = float(hss(eva_pred, eva_obs, thr=thr)['HSS'])
                    eva_mse = float(mse(eva_pred, eva_obs)['MSE'])

                    cc = calculate_cc(eva_pred, eva_obs)
                    acc = calculate_acc(eva_pred, eva_obs, thr)  # Threshold may need adjustment
                    # Append metrics to the respective lists in the dictionary
                    metrics_dict['acc'].append(acc)
                    metrics_dict['eva_csi'].append(eva_csi)
                    metrics_dict['eva_far'].append(eva_far)
                    metrics_dict['eva_hss'].append(eva_hss)
                    metrics_dict['eva_pod'].append(eva_pod)
                    metrics_dict['cc'].append(cc)
                    metrics_dict['eva_mse'].append(eva_mse)

            # Calculate non-NaN mean for each metric
            non_nan_means = {key: np.nanmean(value) for key, value in metrics_dict.items()}

            # Write mean results to the file
            f.write(
                f"{region},{non_nan_means['acc']},{non_nan_means['eva_csi']},{non_nan_means['eva_far']},{non_nan_means['eva_hss']},{non_nan_means['eva_pod']},"
                f"{non_nan_means['cc']},{non_nan_means['eva_mse']},"
                f"{num_p/371.0*100}")
            f.write("\n")
            f.close()


def calculate_metrics_event(pred_total, y_test, configs, scheme, thr):
    shp_root = r".\kw_01_data\china_boundary/"
    shp_list = ['bou1_4p.shp']

    # 使用字典推导式创建字典
    regions = {str(shp_name).split('.shp')[0]: shp_root + shp_name for shp_name in shp_list}
    # 结果评价（4大类方法）
    # 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    # 配置参数
    outcome_path = configs["eval"]["outcome_path"]

    n_leadtimes = 9
    leadt = [j for j in range(n_leadtimes)]
    sample = [i for i in range(len(y_test))]

    # 测试集评价结果的均值
    ## 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    pod = verification.get_method("POD")
    csi = verification.get_method("CSI")
    far = verification.get_method("FAR")
    hss = verification.get_method("HSS")
    mse = verification.get_method("MSE")
    # 调整数据结构，构建对应维度

    pred_total = pred_total[:, 0]
    y_test = y_test[:, 0]

    # 降水数据shape→构建经纬度坐标
    shape = (512, 640)

    # 创建对应的经纬度坐标
    latitudes = np.linspace(2.8, 54, shape[0])
    longitudes = np.linspace(72, 136, shape[1])

    # 创建 xarray 数据集，附加经纬度坐标信息
    pred_xr = xr.DataArray(pred_total, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})
    ture_xr = xr.DataArray(y_test, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})
    pred_xr.rio.set_crs("EPSG:4326", inplace=True)
    ture_xr.rio.set_crs("EPSG:4326", inplace=True)

    # # 保存为 NetCDF 文件（可选）
    # precip_xr.to_netcdf('precip_data.nc')

    # 遍历分区，依次裁剪计算指标
    for key, value in regions.items():
        shp = gpd.read_file(value).set_crs(epsg=4326)
        # shp = gpd.read_file(value).set_crs(epsg=4326)

        # 利用 shapefile 进行裁剪
        clipped_pred = pred_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)
        clipped_ture = ture_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)

        file = outcome_path + f"eval_{scheme}.csv"
        f = open(file, 'a')
        f.write("Region,lead time,ACC,CSI,FAR,HSS,POD,CC,MSE,percent_p(%)")
        f.write("\n")
        for j in range(n_leadtimes):

            # Initialize variables to store cumulative values and counts for each metric
            # Determine the region for the current prediction
            region = key
            # Create dictionaries to store lists for each metric
            num_p = 0
            metrics_dict = {
                'acc': [],
                'eva_csi': [],
                'eva_far': [],
                'eva_hss': [],
                'eva_pod': [],
                'cc': [],
                'eva_mse': [],
            }

            for i in tqdm(range(len(y_test)), desc=f'评价global_mean:[R]{key}_[T]{30 * (j + 1)}min中'):
                eva_pred, eva_obs = clipped_pred.values[i, j, :, :], clipped_ture.values[i, j, :, :]


                if region is not None and np.nanmax(eva_obs) > 0.1:
                    num_p += 1
                    # 各个指标在进行逻辑运算时，会将结果为nan的排除统计
                    eva_pod = float(pod(eva_pred, eva_obs, thr=thr)['POD'])
                    eva_csi = float(csi(eva_pred, eva_obs, thr=thr)['CSI'])
                    eva_far = float(far(eva_pred, eva_obs, thr=thr)['FAR'])
                    eva_hss = float(hss(eva_pred, eva_obs, thr=thr)['HSS'])
                    eva_mse = float(mse(eva_pred, eva_obs)['MSE'])

                    cc = calculate_cc(eva_pred, eva_obs)
                    acc = calculate_acc(eva_pred, eva_obs, thr)  # Threshold may need adjustment
                    # Append metrics to the respective lists in the dictionary
                    metrics_dict['acc'].append(acc)
                    metrics_dict['eva_csi'].append(eva_csi)
                    metrics_dict['eva_far'].append(eva_far)
                    metrics_dict['eva_hss'].append(eva_hss)
                    metrics_dict['eva_pod'].append(eva_pod)
                    metrics_dict['cc'].append(cc)
                    metrics_dict['eva_mse'].append(eva_mse)

            # Calculate non-NaN mean for each metric
            non_nan_means = {key: np.nanmean(value) for key, value in metrics_dict.items()}

            # Write mean results to the file
            f.write(
                f"{region},{30 * (j + 1)},{non_nan_means['acc']},{non_nan_means['eva_csi']},{non_nan_means['eva_far']},{non_nan_means['eva_hss']},{non_nan_means['eva_pod']},"
                f"{non_nan_means['cc']},{non_nan_means['eva_mse']},"
                f"{num_p/371.0*100}")
            f.write("\n")
        f.close()


def calculate_metrics_part_all(pred_total, y_test, configs, scheme, thr_dict):
    # Load shapefiles for each region
    # Convert the coordinate system to WGS84
    shp_root = r".\kw_01_data\china_qu6\part_qu6/"
    shp_list = ['东北.shp', '华北.shp', '华东.shp', '西北.shp', '西南.shp',
                    '中南.shp']

    # 使用字典推导式创建字典
    regions = {str(shp_name).split('.shp')[0]: shp_root + shp_name for shp_name in shp_list}
    # 结果评价（4大类方法）
    # 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    # 配置参数
    outcome_path = configs["eval"]["outcome_path"]
    n_leadtimes = configs['data']['n_leadtimes']

    sal = verification.get_method("SAL")
    # 调整数据结构，构建对应维度
    leadt = [j for j in range(n_leadtimes)]
    sample = [i for i in range(len(y_test))]
    pred_total = pred_total[:, 0]
    y_test = y_test[:, 0]

    # 配置参数
    plot_path = configs["eval"]["fig_path"] + scheme + '/'
    if os.path.exists(plot_path) == False:
        os.makedirs(plot_path)
    n_leadtimes = configs['data']['n_leadtimes']
    metrics = ["POD", "CSI", "FAR", "HSS", "CC", "ACC"]

    # 降水数据shape→构建经纬度坐标
    shape = (512, 640)

    # 创建对应的经纬度坐标
    latitudes = np.linspace(2.8, 54, shape[0])
    longitudes = np.linspace(72, 136, shape[1])

    # 创建 xarray 数据集，附加经纬度坐标信息
    pred_xr = xr.DataArray(pred_total, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})
    ture_xr = xr.DataArray(y_test, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})

    # 附加地理信息
    pred_xr.rio.set_crs("EPSG:4326", inplace=True)
    ture_xr.rio.set_crs("EPSG:4326", inplace=True)
    sal_list = ['SAL(S)', 'SAL(A)', 'SAL(L)']

    # # 保存为 NetCDF 文件（可选）
    # precip_xr.to_netcdf('precip_data.nc')

    # 遍历分区，依次裁剪计算指标
    for key, value in regions.items():
        shp = gpd.read_file(value).to_crs(epsg=4326)
        # 利用 shapefile 进行裁剪
        clipped_pred = pred_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)
        clipped_ture = ture_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)
        for s in sal_list:
            file = outcome_path + f"eval_{scheme}_{key}_{s}.csv"
            f = open(file, 'a')
            f.write("30min,60min,90min")
            f.write("\n")
            for i in tqdm(range(len(y_test)), desc=f'评价[R]{key}_[S]{s}中'):
                for j in range(n_leadtimes):
                    # Initialize variables to store cumulative values and counts for each metric
                    # Determine the region for the current prediction
                    region = key
                    # Create dictionaries to store lists for each metric
                    eva_pred, eva_obs = clipped_pred[i, j, :, :], clipped_ture[i, j, :, :]
                    if region is not None and np.nanmax(eva_obs) > 0.1:
                        try:
                            eva_sal = sal(eva_pred, eva_obs)    #  NaNs are ignored
                        except ZeroDivisionError:
                            print("=" * 70)
                            print(f"scheme:{scheme}, region:{key}, shp:{value}, 分母为0, 无法计算SAL得分")
                            print("=" * 70)
                            eva_sal = [np.nan, np.nan, np.nan]
                        # Write all results to the file
                        if s == 'SAL(S)':
                            f.write(
                                f"{eva_sal[0]},")
                        elif s == 'SAL(A)':
                            f.write(
                                f"{eva_sal[1]},")
                        elif s == 'SAL(L)':
                            f.write(
                                f"{eva_sal[2]},")
                    else:
                        f.write(
                            f"{np.nan},")
                f.write("\n")
            f.write("\n")
            f.close()


def calculate_metrics_part_mean(pred_total, y_test, configs, scheme, thr_dicts):
    # Load shapefiles for each region
    # Convert the coordinate system to WGS84
    shp_root = r".\kw_01_data\china_qu6\part_qu6_English/"
    shp_list = ['NE.shp', 'NC.shp', 'EC.shp', 'NW.shp', 'SW.shp',
                    'SM.shp']

    # 使用字典推导式创建字典
    regions = {str(shp_name).split('.shp')[0]: shp_root + shp_name for shp_name in shp_list}
    # 结果评价（4大类方法）
    # 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    # 配置参数
    outcome_path = configs["eval"]["outcome_path"]

    # 测试集评价结果的均值
    ## 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    pod = verification.get_method("POD")
    csi = verification.get_method("CSI")
    far = verification.get_method("FAR")
    hss = verification.get_method("HSS")
    mse = verification.get_method("MSE")
    pred_total = pred_total[:, 0]
    y_test = y_test[:, 0]

    # 配置参数
    plot_path = configs["eval"]["fig_path"] + scheme + '/'
    if os.path.exists(plot_path) == False:
        os.makedirs(plot_path)
    n_leadtimes = configs['data']['n_leadtimes']
    metrics = ["POD", "CSI", "FAR", "HSS", "CC", "ACC"]

    # 降水数据shape→构建经纬度坐标
    shape = (512, 640)

    # 创建对应的经纬度坐标
    latitudes = np.linspace(2.8, 54, shape[0])
    longitudes = np.linspace(72, 136, shape[1])

    # 创建 xarray 数据集，附加经纬度坐标信息
    pred_xr = xr.DataArray(pred_total, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})
    ture_xr = xr.DataArray(y_test, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'x': longitudes, 'y': latitudes})

    # 附加地理信息
    pred_xr.rio.set_crs("EPSG:4326", inplace=True)
    ture_xr.rio.set_crs("EPSG:4326", inplace=True)
    sal_list = ['SAL(S)', 'SAL(A)', 'SAL(L)']

    # # 保存为 NetCDF 文件（可选）
    # precip_xr.to_netcdf('precip_data.nc')

    # 遍历分区，依次裁剪计算指标
    for thr_p, thr_dict in thr_dicts.items():
        for key, value in regions.items():
            if key in thr_dict.keys():
                thr = float(thr_dict[key])
            else:
                thr = 0.1
            shp = gpd.read_file(value).to_crs(epsg=4326)
            # shp = gpd.read_file(value).set_crs(epsg=4326)

            # 利用 shapefile 进行裁剪
            clipped_pred = pred_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)
            clipped_ture = ture_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)


            for j in range(n_leadtimes):
                file = outcome_path + f"eval_{thr_p}_{scheme}_[thr]{thr}_{30 * (j + 1)}.csv"
                f = open(file, 'a')
                f.write("Region,ACC,CSI,FAR,HSS,POD,CC,MSE,percent_p(%)")
                f.write("\n")

                # Initialize variables to store cumulative values and counts for each metric
                # Determine the region for the current prediction
                region = key
                # Create dictionaries to store lists for each metric
                num_p = 0
                metrics_dict = {
                    'acc': [],
                    'eva_csi': [],
                    'eva_far': [],
                    'eva_hss': [],
                    'eva_pod': [],
                    'cc': [],
                    'eva_mse': [],
                }

                for i in tqdm(range(len(y_test)), desc=f'评价part_mean:[R]{key}_[T]{30 * (j + 1)}min中'):
                    eva_pred, eva_obs = clipped_pred.values[i, j, :, :], clipped_ture.values[i, j, :, :]

                    if region is not None and np.nanmax(eva_obs) > 0.1:
                        num_p += 1
                        # 各个指标在进行逻辑运算时，会将结果为nan的排除统计
                        eva_pod = float(pod(eva_pred, eva_obs, thr=thr)['POD'])
                        eva_csi = float(csi(eva_pred, eva_obs, thr=thr)['CSI'])
                        eva_far = float(far(eva_pred, eva_obs, thr=thr)['FAR'])
                        eva_hss = float(hss(eva_pred, eva_obs, thr=thr)['HSS'])
                        eva_mse = float(mse(eva_pred, eva_obs)['MSE'])
                        cc = calculate_cc(eva_pred, eva_obs)
                        acc = calculate_acc(eva_pred, eva_obs, thr)  # Threshold may need adjustment
                        # Append metrics to the respective lists in the dictionary
                        metrics_dict['acc'].append(acc)
                        metrics_dict['eva_csi'].append(eva_csi)
                        metrics_dict['eva_far'].append(eva_far)
                        metrics_dict['eva_hss'].append(eva_hss)
                        metrics_dict['eva_pod'].append(eva_pod)
                        metrics_dict['cc'].append(cc)
                        metrics_dict['eva_mse'].append(eva_mse)

                # Calculate non-NaN mean for each metric
                non_nan_means = {key: np.nanmean(value) for key, value in metrics_dict.items()}

                # Write mean results to the file
                f.write(
                    f"{region},{non_nan_means['acc']},{non_nan_means['eva_csi']},{non_nan_means['eva_far']},{non_nan_means['eva_hss']},{non_nan_means['eva_pod']},"
                    f"{non_nan_means['cc']},{non_nan_means['eva_mse']},"
                    f"{num_p/371.0*100}")
                f.write("\n")
                f.close()


def calculate_metrics_part_mean_v1(pred_total, y_test, configs, scheme, thr):
    scheme_num = 0  #
    # Load shapefiles for each region
    # Convert the coordinate system to WGS84
    shp_root = r".\kw_01_data\china_qu6\part_qu6/"
    shp_list = ['东北.shp', '华北.shp', '华东.shp', '西北.shp', '西南.shp',
                    '中南.shp']

    # 使用字典推导式创建字典
    regions = {str(shp_name).split('.shp')[0]: shp_root + shp_name for shp_name in shp_list}
    # 配置参数
    outcome_path = configs["eval"]["outcome_path"]
    n_leadtimes = configs['data']['n_leadtimes']

    # 测试集评价结果的均值
    ## 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    pod = verification.get_method("POD")
    csi = verification.get_method("CSI")
    far = verification.get_method("FAR")
    hss = verification.get_method("HSS")
    mse = verification.get_method("MSE")
    sal = verification.get_method("SAL")
    # 调整数据结构，构建对应维度
    leadt = [j for j in range(n_leadtimes)]
    sample = [i for i in range(len(y_test))]
    pred_total = pred_total[:, 0]
    y_test = y_test[:, 0]

    # 配置参数
    plot_path = configs["eval"]["fig_path"] + scheme + '/'
    if os.path.exists(plot_path) == False:
        os.makedirs(plot_path)
    n_leadtimes = configs['data']['n_leadtimes']

    # 降水数据shape→构建经纬度坐标
    shape = (512, 640)

    # 创建对应的经纬度坐标
    latitudes = np.linspace(2.8, 54, shape[0])
    longitudes = np.linspace(72, 136, shape[1])

    # 创建 xarray 数据集，附加经纬度坐标信息
    pred_xr = xr.DataArray(pred_total, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'sample': sample, 'leadt': leadt, 'x': longitudes, 'y': latitudes})
    ture_xr = xr.DataArray(y_test, dims=('sample', 'leadt', 'x', 'y'),
                           coords={'sample': sample, 'leadt': leadt, 'x': longitudes, 'y': latitudes})

    # 附加地理信息
    pred_xr.rio.write_crs("EPSG:4326", inplace=True)
    ture_xr.rio.write_crs("EPSG:4326", inplace=True)

    region_num = 0
    # 遍历分区，依次裁剪计算指标
    for key, value in regions.items():
        region_num += 1
        shp = gpd.read_file(value).to_crs(epsg=4326)
        # shp = gpd.read_file(value).set_crs(epsg=4326)

        # 利用 shapefile 进行裁剪
        clipped_pred = pred_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)
        clipped_ture = ture_xr.rio.clip(shp.geometry.apply(mapping), shp.crs)


        for j in range(n_leadtimes):
            file_path = outcome_path + f"eval_{scheme}_{30 * (j + 1)}.xlsx"
            # Initialize variables to store cumulative values and counts for each metric
            # Determine the region for the current prediction
            region = key
            # Create dictionaries to store lists for each metric
            num_p = 0
            metrics_dict = {
                'POD': [],
                'CSI': [],
                'FAR': [],
                'HSS': [],
                'MSE': [],
                'R-squared': [],
                'CC': [],
                'ACC': []
            }
            sheets = ['POD', 'CSI', 'FAR', 'HSS', 'MSE', 'R-squared', 'ACC', 'percent_p(%)', 'SS']
            if not os.path.exists(file_path):
                # Create a new workbook if it doesn't exist
                for s in range(len(sheets)):
                    df = pd.DataFrame(columns=["model", f"NE_{30 * (j + 1)}", f"NC_{30 * (j + 1)}",
                                               f"EC_{30 * (j + 1)}", f"NW_{30 * (j + 1)}", f"SW_{30 * (j + 1)}",
                                               f"SM_{30 * (j + 1)}", f"NE_{30 * (j + 2)}", f"NC_{30 * (j + 2)}",
                                               f"EC_{30 * (j + 2)}", f"NW_{30 * (j + 2)}", f"SW_{30 * (j + 2)}",
                                               f"SM_{30 * (j + 2)}", f"NE_{30 * (j + 3)}", f"NC_{30 * (j + 3)}",
                                               f"EC_{30 * (j + 3)}", f"NW_{30 * (j + 3)}", f"SW_{30 * (j + 3)}",
                                               f"SM_{30 * (j + 3)}"])
                    try:
                        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
                            df.to_excel(writer, sheet_name=sheets[s], index=False)
                    except:
                        # 保存 DataFrame 到 Excel 文件
                        df.to_excel(file_path, sheet_name=sheets[s], index=False)

            # Initialize variables to store cumulative values and counts for each metric
            for i in range(len(y_test)):
                eva_pred, eva_obs = clipped_pred.values[i, j, :, :], clipped_ture.values[i, j, :, :]
                if region is not None and np.nanmax(eva_obs) > 0.1:
                    num_p += 1
                    eva_pod = float(pod(eva_pred, eva_obs, thr=thr)['POD'])
                    eva_csi = float(csi(eva_pred, eva_obs, thr=thr)['CSI'])
                    eva_far = float(far(eva_pred, eva_obs, thr=thr)['FAR'])
                    eva_hss = float(hss(eva_pred, eva_obs, thr=thr)['HSS'])
                    eva_mse = float(mse(eva_pred, eva_obs)['MSE'])
                    try:
                        eva_sal = sal(eva_pred, eva_obs)
                    except ZeroDivisionError:
                        print("=" * 70)
                        print(f"scheme:{scheme}, region:{key}, shp:{value}, 分母为0, 无法计算SAL得分")
                        print("=" * 70)
                        eva_sal = [np.nan, np.nan, np.nan]

                    # Calculate region-specific metrics
                    r_squared = calculate_r_squared(eva_pred, eva_obs)
                    rbias = calculate_rbias(eva_pred, eva_obs)
                    cc = calculate_cc(eva_pred, eva_obs)
                    acc = calculate_acc(eva_pred, eva_obs, thr)  # Threshold may need adjustment

                    # Append metrics to the respective lists in the dictionary
                    metrics_dict['POD'].append(eva_pod)
                    metrics_dict['CSI'].append(eva_csi)
                    metrics_dict['FAR'].append(eva_far)
                    metrics_dict['HSS'].append(eva_hss)
                    metrics_dict['MSE'].append(eva_mse)
                    metrics_dict['R-squared'].append(r_squared)
                    metrics_dict['ACC'].append(acc)

            non_nan_means = {key: np.nanmean(value) for key, value in metrics_dict.items()}
            for s in range(len(sheets)):
                df = pd.read_excel(file_path, sheet_name=sheets[s])
                # Append mean results to the DataFrame
                df = df._append({"model": scheme, f"NE_{30 * (j + 1)}": non_nan_means[sheets[s]],
                                f"NC_{30 * (j + 1)}": non_nan_means[sheets[s]],
                                f"EC_{30 * (j + 1)}": non_nan_means[sheets[s]],
                                f"NW_{30 * (j + 1)}": non_nan_means[sheets[s]],
                                f"SW_{30 * (j + 1)}": non_nan_means[sheets[s]],
                                f"SM_{30 * (j + 1)}": non_nan_means[sheets[s]],
                                f"NE_{30 * (j + 2)}": non_nan_means[sheets[s]],
                                f"NC_{30 * (j + 2)}": non_nan_means[sheets[s]],
                                f"EC_{30 * (j + 2)}": non_nan_means[sheets[s]],
                                f"NW_{30 * (j + 2)}": non_nan_means[sheets[s]],
                                f"SW_{30 * (j + 2)}": non_nan_means[sheets[s]],
                                f"SM_{30 * (j + 2)}": non_nan_means[sheets[s]],
                                f"NE_{30 * (j + 3)}": non_nan_means[sheets[s]],
                                f"NC_{30 * (j + 3)}": non_nan_means[sheets[s]],
                                f"EC_{30 * (j + 3)}": non_nan_means[sheets[s]],
                                f"NW_{30 * (j + 3)}": non_nan_means[sheets[s]],
                                f"SW_{30 * (j + 3)}": non_nan_means[sheets[s]],
                                f"SM_{30 * (j + 3)}": non_nan_means[sheets[s]]}, ignore_index=True)
                # 使用ExcelWriter将数据写入到指定的工作表
                # 使用 openpyxl 打开 Excel 文件
                with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
                    # 加载现有的 Excel 工作簿
                    # 选择指定的 sheet
                    sheet = writer.book[sheets[s]]

                    # 将 DataFrame 数据追加到 sheet 的下一行
                    for index, row in df.iterrows():
                        sheet.append(row.tolist())
                    # 保存更改
                    writer.book.save(file_path)


def plot_precip(data, n_leadtimes, path):
    for i in range(len(data)):
        # 预测结果可视化
        for j in range(n_leadtimes):
            plt.figure(figsize=(9, 5), dpi=800)
            precip_2d = data[i, :, j, :, :].reshape(640, 512).T
            plot_precip_field(precip_2d, axis="off")
            plt.savefig(path + 'fig_第%d个样本第%dmin预见期_pre.jpg' % (i, (j + 1) * 30))
            plt.close()


def cal_plot_metrics(pred_total, y_test, configs, test_outname, thr):
    # 结果评价（4大类方法）
    # 1.nowcasting skill scores:(POD,CSI,FAR,HSS,FSS)
    # 配置参数
    pred_total = pred_total[:, 0]
    y_test = y_test[:, 0]
    # 配置参数
    plot_path = configs["eval"]["fig_path"] + test_outname + '/'
    if os.path.exists(plot_path) == False:
        os.makedirs(plot_path)
    n_leadtimes = configs['data']['n_leadtimes']

    # 测试集评价结果的均值
    metrics = ["POD", "CSI", "FAR", "HSS", "CC", "ACC"]
    # 初始化结果数组
    result_arrays = {metric: np.zeros((3, 640, 512)) for metric in metrics}
    eva_list1 = ["POD", "CSI", "FAR", "HSS"]
    eva_list3 = ["CC"]
    # 循环计算指标

    for lon in range(y_test.shape[2]):
        for lat in range(y_test.shape[3]):
            for metric in tqdm(metrics, desc=f'calculate_global_for_plot_[lon:{lon},lat:{lat}]'):
                for j in range(n_leadtimes):
                    point_observed, point_predicted = y_test[:, j, lon, lat], pred_total[:, j, lon, lat]
                    eva_outcome = None
                    if metric in eva_list1:
                        method = verification.get_method(metric)
                        eva_outcome = float(method(point_predicted, point_observed, thr=thr)[metric])
                    elif metric in eva_list3:
                        method = None
                        if metric == "R-squared":
                            method = calculate_r_squared
                        elif metric == "RBIAS":
                            method = calculate_rbias
                        elif metric == "CC":
                            method = calculate_cc
                        eva_outcome = method(point_predicted, point_observed)
                    elif metric == "ACC":
                        eva_outcome = calculate_acc(point_predicted, point_observed, thr)
                    result_arrays[metric][j, lon, lat] = eva_outcome
                print(f"lon:{lon},lat:{lat},metric:{metric}------over")
    shp_file_path = configs['data']['shp_file_path']
    plot_eva_tiff(result_arrays, n_leadtimes, plot_path, shp_file_path, *metrics)


def iterate_prediction(model, data_history, iterate_times=3):
    """
    Function to iteratively predict future time steps using a trained deep learning model.

    Parameters:
    - model: 已经训练好的深度学习模型
    - history: 历史的9个时刻的降水图，torch格式的张量，shape为(9, C, H, W)，C为通道数，H为高度，W为宽度
    - future_steps: 需要预测的未来时间步数，默认为3

    Returns:
    - predictions: 预测的未来时刻的降水图，torch格式的张量，shape为(future_steps, C, H, W)
    """
    with torch.no_grad():
        model.eval()
        # 遍历每一个未来时间步
        for step in tqdm(range(iterate_times), desc='Iterative prediction:'):
            # 获取历史6个时刻的输入数据
            input_data = data_history[:, :, -9:, :, :]

            # 使用模型进行预测
            output = model(input_data)

            # 更新历史数据，将当前预测结果加入历史数据中
            data_history = torch.cat((data_history, output), dim=2)
    iter_out = data_history[:, :, -9:, :, :].cpu().numpy()  # 将张量从计算图中分离出来并转化为numpy array
    iter_out[iter_out < 0] = 0
    # 将预测结果列表转换为张量并返回
    return iter_out


def main(model_name, model_PSU, run_pattern, model_file_t):
    configs = json.load(open('config_NcN.json', 'r', encoding='utf-8', errors='ignore'))  # 加载模拟配置json
    n_leadtimes = configs['data']['n_leadtimes']
    # 1、数据：打开处理划分
    file_path = configs["data"]["file_path"]
    model_path = configs["model"]["model_path"]
    data_gpm = Data_GPM(configs)
    data_len = configs["data"]["data_len"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_list = os.listdir(file_path)

    model = create_model(model_name, model_PSU, device)

    # 在这里添加训练代码，包括数据加载、损失函数、优化器等
    # 2、模型：训练验证优化
    scheme_name = 'IP%s_[M]%s-[OF]%s' % (file_path.split('0.')[-1].split('selected')[0], model_name, model_PSU)
    # Different threshold for evaluation
    thr_dicts = configs["eval"]["thr_precs"]

    if run_pattern == 'new_train':
        print(model)
        # 实例化 EarlyStopping 类
        early_stopping = EarlyStopping(patience=5, verbose=True)
        lr = configs["training"]["learning rate"]
        # 定义损失函数和优化器
        criterion = nn.SmoothL1Loss()
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # 训练循环
        num_epochs = configs['training']['epochs']
        batch_size = configs['training']['batch_size']

        # 每隔Te个epoch循环一次lr（即lr更新的周期），每个batch更新lr时的步长tt
        training_config = configs["training"]
        Estart, Te, tt, multFactor, t0, TeNext = training_config['Estart'], training_config['Te'], training_config[
            'tt'], training_config['multFactor'], math.pi / 2.0, training_config['Te']

        t_allstart = t.time()  # 记录训练开始前时间
        for epoch in range(num_epochs):
            t_start = t.time()  # 记录每个epoch开始前时间
            model.train()  # 设置模型为训练模式
            train_loss = 0.0
            # 清零梯度
            optimizer.zero_grad()
            # 按一定数目遍历文件夹迭代训练model
            kiwi = 0
            file_group = split_list(file_list, data_len)
            for file_n in file_group:
                data_gpm.load_data(file_n)  # 调取长度为18的数据进行分析
                x_train, y_train = data_gpm.data, data_gpm.label
                # x_train, y_train = data_gpm.clip_quarter(x_train), data_gpm.clip_quarter(y_train) # 裁剪为1/4进行训练
                # 准备训练数据和验证数据（转换成张张量tensor）
                try:
                    x_train = torch.from_numpy(x_train).to(device)
                    y_train = torch.from_numpy(y_train).to(device)
                except:
                    print(f"x_type:{type(x_train)},x_data:{x_train}")
                    print(f"y_type:{type(y_train)},y_data:{y_train}")
                # 增加一个通道数的维度，用于卷积输入
                x_train, y_train = x_train[:, np.newaxis, :, :, :], y_train[:, np.newaxis, ::, :]

                # 创建数据加载器
                data_train = Data(x_train, y_train)
                DataLoader_train = DataLoader(data_train, batch_size=batch_size,
                                              shuffle=False)  # Dataloader没有数值索引→enumerate
                # 前向传播
                # 迭代索引访问批次数据
                for batch_index, batch_data in enumerate(DataLoader_train):
                    batch_x, batch_y = batch_data
                    optimizer.zero_grad()
                    outputs = model(batch_x.to(device))
                    loss = criterion(outputs, batch_y)
                    kiwi += 3 * 18
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    train_loss = loss.item()
                    if configs["training"]["adjust_lr"] == True:
                        # 调整学习率
                        adjust_learning_rate(configs, epoch, optimizer, lr, (Estart, Te, tt, multFactor, t0))
                    print("kiwi:", kiwi)
                    print(f"train loss: {train_loss: .2f}")
                    print('-' * 50)
            # 重新设置周期
            if (epoch + 1 == TeNext):  # time to restart
                tt = 0  # by setting to 0 we set lr to lr_max, see above
                Te = Te * multFactor  # change the period of restarts
                TeNext = TeNext + Te  # note the next restart's epoch
                # 训练结束后，你可以保存模型的参数
                # 设置模型保存路径

            model.eval()
            # 使用 EarlyStopping 判断是否停止训练
            # 加载verif数据进行测试并输出对应结果
            x_verif, y_verif = data_gpm.load_verif(5346)
            # x_verif, y_verif = data_gpm.clip_quarter(x_verif), data_gpm.clip_quarter(y_verif)
            x_verif = torch.from_numpy(x_verif).to(device)
            # 增加一个通道数的维度，用于卷积输入
            x_verif, y_verif = x_verif[:, np.newaxis, :, :, :], y_verif[:, np.newaxis, :, :, :]
            # 创建数据加载器
            data_verif = Data(x_verif, y_verif)
            DataLoader_verif = DataLoader(data_verif, batch_size=batch_size,
                                          shuffle=False)

            val_loss = 0.0
            total_samples = 0

            with torch.no_grad():  # 禁用梯度计算
                for inputs, targets in DataLoader_verif:  # 假设 val_loader 是你的验证集 DataLoader
                    outputs = model(inputs)
                    loss = criterion(outputs.to(device), targets.to(device))
                    val_loss += loss.item() * len(targets)
                    total_samples += len(targets)

            val_loss /= total_samples
            early_stopping(val_loss, model)

            if epoch + 1 == 70:     # early_stopping.early_stop:
                print("Early stopping")
                save_fname = os.path.join(configs['training']['save_dir'], scheme_name + '--%s_e%s-%s_[Vl]%f__verif.h5' %
                                          (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), epoch + 1,
                                           str(num_epochs), val_loss))
                # 保存模型及其他相关信息

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, save_fname)
                break

            t_epoch = t.time()
            # 打印训练和验证损失
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, time_cost:{(t_epoch - t_start):4f}")
        t_allepochs = t.time()
        print(f"======Total_time_cost:{(t_allepochs - t_allstart) / 60:4f} min======")

    elif run_pattern == 'continue_train':
        model_file = input('请输入需要继续训练的模型文件(.h5)：')
        checkpoint = torch.load(model_path + model_file)
        # 训练循环
        num_epochs = configs['training']['epochs']
        batch_size = configs['training']['batch_size']

        # 每隔Te个epoch循环一次lr（即lr更新的周期），每个batch更新lr时的步长tt
        training_config = configs["training"]

        Estart, Te, tt, multFactor, t0, TeNext = training_config['Estart'], training_config['Te'], training_config[
            'tt'], training_config['multFactor'], math.pi / 2.0, training_config['Te']
        # 定义损失函数和优化器
        criterion = nn.SmoothL1Loss()
        lr = configs["training"]["learning rate"]
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # 步骤3：加载模型和优化器的状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        t_allstart = t.time()  # 记录训练开始前时间
        for epoch in range(epoch_start, num_epochs):
            t_start = t.time()  # 记录每个epoch开始前时间
            model.train()  # 设置模型为训练模式
            train_loss = 0.0

            # 按一定数目遍历文件夹迭代训练model
            kiwi = 0
            file_group = split_list(file_list, data_len)
            for file_n in file_group:
                data_gpm.load_data(file_n)  # 调取长度为18的数据进行分析
                x_train, y_train = data_gpm.data, data_gpm.label

                # 准备训练数据和验证数据（转换成张张量ensor）
                try:
                    x_train = torch.from_numpy(x_train).to(device)
                    y_train = torch.from_numpy(y_train).to(device)
                except:
                    print(f"x_type:{type(x_train)},x_data:{x_train}")
                    print(f"y_type:{type(y_train)},y_data:{y_train}")
                # 增加一个通道数的维度，用于卷积输入
                x_train, y_train = x_train[:, np.newaxis, :, :, :], y_train[:, np.newaxis, ::, :]

                # 创建数据加载器
                data_train = Data(x_train, y_train)
                DataLoader_train = DataLoader(data_train, batch_size=batch_size,
                                              shuffle=False)  # Dataloader没有数值索引→enumerate
                # 前向传播
                # 迭代索引访问批次数据
                for batch_index, batch_data in enumerate(DataLoader_train):
                    batch_x, batch_y = batch_data
                    optimizer.zero_grad()
                    outputs = model(batch_x.to(device))
                    loss = criterion(outputs, batch_y)
                    kiwi += 3 * 18
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    train_loss = loss.item()
                    if configs["training"]["adjust_lr"] == True:
                        # 调整学习率
                        adjust_learning_rate(configs, epoch, optimizer, lr, (Estart, Te, tt, multFactor, t0))
                    print("kiwi:", kiwi)
                    print(f"train loss: {train_loss: .2f}")
                    print('-' * 50)
            # 重新设置周期
            if (epoch + 1 == epoch_start + TeNext + 1):  # time to restart
                tt = 0  # by setting to 0 we set lr to lr_max, see above
                Te = Te * multFactor  # change the period of restarts
                TeNext = TeNext + Te  # note the next restart's epoch
                # 训练结束后，你可以保存模型的参数
                # 设置模型保存路径

                save_fname = os.path.join(configs['training']['save_dir'], scheme_name + '--%s_e%s-%s.h5' %
                                          (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), epoch + 1,
                                           str(num_epochs)))
                # 保存模型及其他相关信息
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, save_fname)
                # 加载verif数据进行测试并输出对应结果
                x_verif, y_verif = data_gpm.load_verif(5346)
                x_verif = torch.from_numpy(x_verif).to(device)
                # 增加一个通道数的维度，用于卷积输入
                x_verif, y_verif = x_verif[:, np.newaxis, :, :, :], y_verif[:, np.newaxis, :, :, :]
                model.eval()
                pred_total = predicted_results(model, x_verif)
                verif_outname = f"verif_{scheme_name}_{epoch + 1}-{str(num_epochs)}"
                calculate_metrics_all(pred_total, y_verif, configs, verif_outname)

            t_epoch = t.time()
            # 打印训练和验证损失
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, time_cost:{(t_epoch - t_start):4f}")
        t_allepochs = t.time()
        print(f"======Total_time_cost:{(t_allepochs - t_allstart) / 60:4f} min======")

    elif run_pattern == 'model_test_global':
        model_file = model_file_t  # input('请输入需要测试的模型文件(.h5)：')
        model_epoch = model_file.split('_')[-1].split('.h5')[0]
        try:
            model.load_state_dict(torch.load(model_path + model_file))  # ['model_state_dict']
        except:
            model.load_state_dict(torch.load(model_path + model_file)['model_state_dict'])  # ['model_state_dict']
        print(model)

        # 加载test数据进行测试并输出对应结果
        # 3、结果：评估绘图分析
        x_test, y_test = data_gpm.load_test(6678)
        x_test = torch.from_numpy(x_test).to(device)
        # 增加一个通道数的维度，用于卷积输入
        x_test, y_test = x_test[:, np.newaxis, :, :, :], y_test[:, np.newaxis, :, :, :]
        model.eval()
        pred_total = predicted_results(model, x_test)
        thr = thr_dicts['china']
        test_outname = f"test-global_{scheme_name}_{model_epoch}_[thr]{thr}"
        calculate_metrics_all(pred_total, y_test, configs, test_outname, thr)
        #calculate_metrics_mean(pred_total, y_test, configs, test_outname, thr)
        cal_plot_metrics(pred_total, y_test, configs, test_outname, thr)

    elif run_pattern == 'model_test_part':
        # model_file = input('请输入需要测试的模型文件(.h5)：')
        model_file = model_file_t
        model_epoch = model_file.split('_')[-1].split('.h5')[0]
        try:
            model.load_state_dict(torch.load(model_path + model_file))  # ['model_state_dict']
        except:
            model.load_state_dict(torch.load(model_path + model_file)['model_state_dict'])  # ['model_state_dict']
        print(model)

        # 加载test数据进行测试并输出对应结果
        # 3、结果：评估绘图分析
        x_test, y_test = data_gpm.load_test(6678)  # 6678
        x_test = torch.from_numpy(x_test).to(device)
        # 增加一个通道数的维度，用于卷积输入
        x_test, y_test = x_test[:, np.newaxis, :, :, :], y_test[:, np.newaxis, :, :, :]
        model.eval()
        pred_total = predicted_results(model, x_test)
        test_outname = f"test-plot_{scheme_name}_{model_epoch}"

        # calculate_metrics_part_all(pred_total, y_test, configs, test_outname, thr_dict)
        calculate_metrics_part_mean(pred_total, y_test, configs, test_outname, thr_dicts)

    elif run_pattern == 'model_event_plot':
        # model_file = input('请输入需要测试的模型文件(.h5)：')
        model_file = model_file_t
        model_epoch = model_file.split('_')[-1].split('.h5')[0]
        try:
            model.load_state_dict(torch.load(model_path + model_file))  # ['model_state_dict']
        except:
            model.load_state_dict(torch.load(model_path + model_file)['model_state_dict'])  # ['model_state_dict']
        print(model)

        # 加载test数据进行测试并输出对应结果
        # 3、结果：评估绘图分析
        events_dir = ['event_2022_0903_0730-0830', 'event_2023_0418_2230-2330', 'event_2023_0729_0300-0400']
        for event_dir in tqdm(events_dir, desc='plotting the predicted precipitation events：'):
            event_path = r'.\kw_01_data\data_20200101_20230531\try1_china_TVT\A_TVT_0.6selected_IP\d_classic_event/' \
                         + f'{event_dir}/'

            x_test, y_test = data_gpm.load_event(18, event_path)  # 6678
            x_test = torch.from_numpy(x_test).to(device)
            # 增加一个通道数的维度，用于卷积输入
            x_test, y_test = x_test[:, np.newaxis, :, :, :], y_test[:, np.newaxis, :, :, :]
            model.eval()
            pred_total = predicted_results(model, x_test)[:, 0, :, :, :]
            for j in tqdm(range(n_leadtimes), desc="plotting predicted precipitation event："):
                pred_event = pred_total[0, j, :, :].T
                # 将小于 0.1 的值替换为 NaN-->绘图时不进行展示
                pred_event[pred_event < 0.1] = 0    # np.nan
                event_figpath = f'.\kw_05_plot/test_classic_event/{event_dir}'
                if os.path.exists(event_figpath) == False:
                    os.makedirs(event_figpath)
                event_figname = event_figpath + f"/test-plot_{scheme_name}_{model_epoch}_{(j+1)*30}min.jpg"
                # 1. 读取并绘制shp文件
                shp_file_path = '.\kw_01_data\china_boundary/bou1_4p.shp'
                china_boundary = gpd.read_file(shp_file_path)
                # 2. 绘制图形
                fig, ax = plt.subplots(
                    subplot_kw={'projection': ccrs.PlateCarree()})  # ax.set_extent([73, 136, 18, 54])  # 设置中国的经纬度范围
                # 添加国界
                china_feature = ShapelyFeature(china_boundary.geometry, ccrs.PlateCarree(),
                                               edgecolor='k', facecolor='none')
                ax.add_feature(china_feature)
                plot_precip_field(pred_event, geodata={"x1": 72, "x2": 136, "y1": 2.8, "y2": 54, "yorigin": "lower"}, ax=ax,
                                  axis="off")
                plt.title(f'{scheme_name}')
                plt.savefig(event_figname)
                plt.close()

    elif run_pattern == 'iterate_event_plot':
        model_file = model_file_t  # input('请输入需要测试的模型文件(.h5)：')
        model_epoch = model_file.split('_')[-1].split('.h5')[0]
        try:
            model.load_state_dict(torch.load(model_path + model_file))  # ['model_state_dict']
        except:
            model.load_state_dict(torch.load(model_path + model_file)['model_state_dict'])  # ['model_state_dict']
        print(model)
        thr = 0.1
        test_outname = f"iterate_test_event_{scheme_name}_{model_epoch}_[thr]{thr}"
        # 绘制典型事件的降水预测图
        events_dir = ['event_2022_0903_0730-0830', 'event_2023_0418_2230-2330', 'event_2023_0729_0300-0400']
        for event_dir in tqdm(events_dir, desc='plotting the predicted precipitation events：'):
            event_path = r'.\kw_01_data\data_20200101_20230531\try1_china_TVT\A_TVT_0.6selected_IP\d_classic_event/' \
                         + f'{event_dir}/'

            x_test, y_test = data_gpm.load_event(18, event_path)  # 6678
            x_test = torch.from_numpy(x_test).to(device)
            # 增加一个通道数的维度，用于卷积输入
            x_test, y_test = x_test[:, np.newaxis, :, :, :], y_test[:, np.newaxis, :, :, :]     # (371, 0 , 18, 640,512)

            model.eval()
            iterate_times = 3
            iterate_out = iterate_prediction(model, x_test, iterate_times)
            for j in tqdm(range(n_leadtimes*iterate_times), desc="plotting predicted precipitation event："):
                pred_event = iterate_out[0, 0, j, :, :].T
                # 将小于 0.1 的值替换为 NaN-->绘图时不进行展示
                pred_event[pred_event < 0.1] = 0    # np.nan
                event_figpath = f'.\kw_05_plot/test_classic_event/event_iterate_test/{event_dir}'
                if os.path.exists(event_figpath) == False:
                    os.makedirs(event_figpath)
                event_figname = event_figpath + f"/test-plot_{scheme_name}_{model_epoch}_{(j+1)*30}min.jpg"
                # 1. 读取并绘制shp文件
                shp_file_path = '.\kw_01_data\china_boundary/bou1_4p.shp'
                china_boundary = gpd.read_file(shp_file_path)
                # 2. 绘制图形
                fig, ax = plt.subplots(
                    subplot_kw={'projection': ccrs.PlateCarree()})  # ax.set_extent([73, 136, 18, 54])  # 设置中国的经纬度范围
                # 添加国界
                china_feature = ShapelyFeature(china_boundary.geometry, ccrs.PlateCarree(),
                                               edgecolor='k', facecolor='none')
                ax.add_feature(china_feature)
                plot_precip_field(pred_event, geodata={"x1": 72, "x2": 136, "y1": 2.8, "y2": 54, "yorigin": "lower"}, ax=ax,
                                  axis="off")
                plt.title(f'{scheme_name.split("6_")[1]}')
                plt.savefig(event_figname)
                plt.close()

    elif run_pattern == 'iterate_event_eval':
        model_file = model_file_t  # input('请输入需要测试的模型文件(.h5)：')
        model_epoch = model_file.split('_')[-1].split('.h5')[0]
        try:
            model.load_state_dict(torch.load(model_path + model_file))  # ['model_state_dict']
        except:
            model.load_state_dict(torch.load(model_path + model_file)['model_state_dict'])  # ['model_state_dict']
        print(model)
        thr = 0.1
        test_outname = f"iterate_test_event_{scheme_name}_{model_epoch}_[thr]{thr}"
        # 绘制典型事件的降水预测图
        events_dir = ['event_2023_0729_0300-0400']
        for event_dir in tqdm(events_dir, desc='plotting the predicted precipitation events：'):
            event_path = r'.\kw_01_data\data_20200101_20230531\try1_china_TVT\A_TVT_0.6selected_IP\d_classic_event/' \
                         + f'{event_dir}/'

            x_test, y_test = data_gpm.load_event(18, event_path)  # 6678
            x_test = torch.from_numpy(x_test).to(device)
            # 增加一个通道数的维度，用于卷积输入
            x_test, y_test = x_test[:, np.newaxis, :, :, :], y_test[:, np.newaxis, :, :, :]     # (371, 0 , 18, 640,512)

            model.eval()
            iterate_times = 3
            iterate_out = iterate_prediction(model, x_test, iterate_times)
            calculate_metrics_event(iterate_out, y_test, configs, test_outname, thr)


    else:
        print("The model pattern you choose is wrong, please choose again.")


def multi_model_run(model_list, OF_list, model_test_list, run_pattern):
    model_t_tested = 0
    model_t_error = 0
    model_error_l = []
    for m in model_list:
        for of in OF_list:
            model_name = m
            model_PSU = of
            # main(model_name, model_PSU, run_pattern, None)
            model_file_t = re.compile(f"IP6_\\[M\\]{m}-\\[OF\\]{of}--(.*)_e70-100.h5")
            for model_t in model_test_list:
                match = model_file_t.match(model_t)
                if match:
                    # try:
                    main(model_name, model_PSU, run_pattern, model_t)
                    model_t_tested += 1
                    # except:
                    #     print('model test error:', model_t)
                    #     model_error_l.append(model_t)
                    #     model_t_error += 1
                    #     continue
    print("=" * 70)
    print(f"After the program runs, the model test is: Success[{model_t_tested}], Error[{model_t_error}]")
    print(model_error_l)
    print("=" * 70)


if __name__ == '__main__':
    model_list = ['CNcN', 'RNcN']  # 'CNcN', 'RNcN'
    OF_list = ['false', 'PSU', 'PSU_ONE', 'CMN', 'RMN']  # 'fc'c', 'PSU', 'PSU_ONE', 'CMN', 'RMN'
    model_test_list = os.listdir(".\kw_03_saved_models")
    run_pattern = 'iterate_event_plot'
    # Operate on all model policies
    multi_model_run(model_list, OF_list, model_test_list, run_pattern)
    sys.exit()