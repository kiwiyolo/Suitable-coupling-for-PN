# -*- coding: UTF-8 -*-
#
# Copyright (C) 2023 - 2023 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2023/9/13 12:25
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : data_out-display_shp&prec
# @IDE     : PyCharm
# -----------------------------------------------------------------
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from netCDF4 import Dataset
from pysteps.visualization import plot_precip_field
import os

# 获取当前脚本文件所在的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 构建上一级文件夹的路径
parent_path = os.path.dirname(current_path)

def plot_precip(data, n_leadtimes, path):
    for i in range(len(data)):
        # 预测结果可视化
        for j in range(n_leadtimes):
            plt.figure(figsize=(9, 5), dpi=800)
            precip_2d = data[i, :, j, :, :].reshape(640, 512).T
            plot_precip_field(precip_2d, axis="off")
            plt.savefig(path + 'fig_第%d个样本第%dmin预见期_pre.jpg' % (i, (j + 1) * 30))
            plt.close()

def plot_precip_shp(nc_files_path, ncfile, fig_name):

    save_name = parent_path + r'\kw_05_plot\test_classic_event/'+ fig_name
    # 1. 读取并绘制shp文件
    shp_file_path = parent_path + r'\kw_01_data\china_boundary/bou1_4p.shp'
    china_boundary = gpd.read_file(shp_file_path)

    # 2. 读取nc.4文件
    nc_file_path = nc_files_path + '/' + ncfile
    nc_data = Dataset(nc_file_path, 'r')
    # 假设nc文件中的降水数据是二维数组，并有纬度和经度数据
    lats = nc_data.variables['lat'][:].data
    lons = nc_data.variables['lon'][:].data
    precipitation = nc_data.variables['precipitationCal'][0, :, :].data.T


    # 3. 绘制图形
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})   # ax.set_extent([73, 136, 18, 54])  # 设置中国的经纬度范围


    # 添加国界
    china_feature = ShapelyFeature(china_boundary.geometry, ccrs.PlateCarree(),
                                   edgecolor='k', facecolor='none')
    ax.add_feature(china_feature)
    plot_precip_field(precipitation, geodata={"x1":72, "x2":136, "y1":2.8, "y2":54,"yorigin": "lower"}, ax=ax, axis="off")
    plt.title('Ground truth')
    plt.savefig(save_name)
    nc_data.close()

def main():

    nc_files_path = parent_path + f'\kw_01_data\data_20200101_20230531/try1_china_TVT\A_TVT_0.6selected_IP\d_classic_event'
    file_i = 0
    for dir_name in os.listdir(nc_files_path):
        file_path = nc_files_path + '/' + dir_name
        for file in os.listdir(file_path):
            file_i += 1
            if file.endswith("nc4") :   # and file_i>9 and file_i<13只展示预测的提前期gt
                fig_name = file.split('.')[0] + '.jpg'
                plot_precip_shp(file_path, file, fig_name)
            if file_i == 18:
                file_i = 0


if __name__ == '__main__':
    main()
    sys.exit()