# -*- coding: UTF-8 -*-
#
# Copyright (C) 2023 - 2023 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2023/12/28 13:37
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : plot_multi_prec-eval
# @IDE     : PyCharm
# -----------------------------------------------------------------

import os
import netCDF4 as nc
import matplotlib.pyplot as plt
from pysteps.visualization import plot_precip_field
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib import cm, colors
import numpy as np
from cartopy.feature import ShapelyFeature
import shapely.geometry as geometry
from tqdm import tqdm
from osgeo import gdal, osr

# 定义函数来加载 nc4 文件并绘制图像
def _plot_precip_from_nc(file_path, ax, china_boundary):
    # 打开 nc4 文件
    dataset = nc.Dataset(file_path, 'r')

    # 从 nc4 文件中获取降水数据，你需要根据实际数据的变量名进行调整
    precip_data = dataset.variables['precipitationCal'][0, :, :].data.T  # 假设变量名为 'precipitation'
    china_feature = ShapelyFeature(china_boundary.geometry, ccrs.PlateCarree(),
                                   edgecolor='k', facecolor='none')
    ax.add_feature(china_feature)
    # 使用 pysteps 的 plot_precip_field 函数绘制降水图
    plot_precip_field(precip_data,geodata={"x1":73, "x2":136, "y1":18, "y2":54,"yorigin": "upper"}, ax=ax, axis="off",colorbar=False)


def plot_eva_study_original(data, n_leadtimes, path, shp_file_path, *metrics):
    # 1. 读取并绘制shp文件
    china_boundary = gpd.read_file(shp_file_path)

    for metric in metrics:
        for j in range(n_leadtimes):
            fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})

            # 提取数据集
            dataset_2d = data[metric][j, :, :].T
            # dataset_2d = data[metric][j, :, :]

            # 添加中国边界
            china_feature = ShapelyFeature(china_boundary.geometry, ccrs.PlateCarree(),
                                           edgecolor='k', facecolor='none')
            ax.add_feature(china_feature)

            # 绘制数据集
            extent = [72, 136, 2.8, 54]  # x1, x2, y1, y2
            img = ax.imshow(dataset_2d, cmap='viridis', extent=extent, origin='lower',
                            transform=ccrs.PlateCarree())
            # 添加等差 colorbar，设置colorbar长度与x轴一致
            box = ax.get_position()
            cax = plt.axes((box.x0, box.y0 - 0.065, box.width, 0.015))
            plt.colorbar(img, cax=cax, orientation='horizontal')
            cax.set_xlabel(metric, fontsize=18)  # 设置colorbar标签并指定字体大小

            # 设置标题
            ax.set_title(f"{metric}_{(j + 1) * 30}", fontsize=20)  # 设置标题并指定字体大小

            # 添加经纬度坐标轴
            ax.set_xticks(range(80, 136, 10), crs=ccrs.PlateCarree())
            ax.set_yticks(range(8, 54, 10), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}$^\circ$E'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}$^\circ$N'.format(y)))

            # 添加横轴和纵轴标签
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Latitude', fontsize=16)


            # 设置坐标轴标签的字体大小
            ax.tick_params(axis='both', labelsize=14)

            # 保存图像
            plt.savefig(f"{path}fig_test_{(j + 1) * 30}min预见期_eva-{metric}.jpg")
            plt.close()


def plot_eva_study(data, n_leadtimes, path, shp_file_path, *metrics):
    # 1. 读取并绘制shp文件
    china_boundary = gpd.read_file(shp_file_path)

    for metric in metrics:
        for j in range(n_leadtimes):
            fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})

            # 提取数据集
            dataset_2d = data[metric][j, :, :].T

            # 根据shp范围创建shapely几何对象
            boundary_polygon = china_boundary.geometry.unary_union
            shape = (512, 640)
            # 创建对应的经纬度坐标
            latitudes = np.linspace(2.8, 54, shape[0])
            longitudes = np.linspace(72, 136, shape[1])
            # 根据几何对象的边界框创建网格
            min_x, min_y, max_x, max_y = boundary_polygon.bounds
            extent = [min_x, max_x, min_y, max_y]

            # 创建一个meshgrid用于检查每个点是否在范围内
            x, y = np.meshgrid(longitudes,latitudes)
            points = np.column_stack((x.ravel(), y.ravel()))


            # 检查每个点是否在shp范围内
            for pt in tqdm(points, desc='detect if the point is in shp:'):
                if not geometry.Point(pt).within(geometry.shape(boundary_polygon)):
                # 将范围之外的值设为NaN
                    lat = np.where(latitudes==pt[1])[:]
                    lon = np.where(longitudes==pt[0])[:]
                    dataset_2d[lat, lon] = np.nan

            # 添加中国边界
            china_feature = ShapelyFeature(china_boundary.geometry, ccrs.PlateCarree(),
                                           edgecolor='k', facecolor='none')
            ax.add_feature(china_feature)

            # 绘制数据集
            img = ax.imshow(dataset_2d, cmap='viridis', extent=extent, origin='lower',
                            transform=ccrs.PlateCarree())
            # 添加等差 colorbar，设置colorbar长度与x轴一致
            box = ax.get_position()
            cax = plt.axes((box.x0, box.y0 - 0.065, box.width, 0.015))
            plt.colorbar(img, cax=cax, orientation='horizontal')
            cax.set_xlabel(metric, fontsize=18)  # 设置colorbar标签并指定字体大小

            # 设置标题
            ax.set_title(f"{metric}_{(j + 1) * 30}", fontsize=20)  # 设置标题并指定字体大小

            # 添加经纬度坐标轴
            ax.set_xticks(range(80, 136, 10), crs=ccrs.PlateCarree())
            ax.set_yticks(range(8, 54, 10), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}$^\circ$E'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}$^\circ$N'.format(y)))

            # 添加横轴和纵轴标签
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Latitude', fontsize=16)


            # 设置坐标轴标签的字体大小
            ax.tick_params(axis='both', labelsize=14)

            # 保存图像
            plt.savefig(f"{path}fig_test_{(j + 1) * 30}min预见期_eva-{metric}.jpg")
            plt.close()


def plot_eva_tiff(data, n_leadtimes, path, shp_file_path, *metrics):
    # 1. 读取并绘制shp文件
    china_boundary = gpd.read_file(shp_file_path)

    for metric in tqdm(metrics, desc='saving evaluation output:'):
        for j in range(n_leadtimes):
            fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})

            # 提取数据集
            dataset_2d = data[metric][j, :, :].T

            # 添加中国边界
            china_feature = ShapelyFeature(china_boundary.geometry, ccrs.PlateCarree(),
                                           edgecolor='k', facecolor='none')
            ax.add_feature(china_feature)

            # 绘制数据集
            extent = [72.0, 136.0, 2.8, 54.0]  # x1, x2, y1, y2

            # 保存为tiff文件
            save_as_tiff(dataset_2d, extent, f"{path}fig_test_{(j + 1) * 30}min预见期_eva-{metric}.tif")

def save_as_tiff(data, extent, filename):
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = data.shape
        ds = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)

        # 设定投影信息
        # 左上角X坐标、水平像素分辨率、旋转参数、左上角Y坐标、旋转参数和垂直像素分辨率
        ds.SetGeoTransform(
            [extent[0], (extent[1] - extent[0]) / float(cols), 0, extent[2], 0, (extent[3] - extent[2]) / float(rows)])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # EPSG code for WGS84
        # ds.SetProjection(srs.ExportToWkt())

        # 写入数据
        band = ds.GetRasterBand(1)
        band.WriteArray(data)

        # 关闭文件
        ds = None

def _dynamic_formatting_floats(float_array, colorscale="pysteps"):
    """Function to format the floats defining the class limits of the colorbar."""
    float_array = np.array(float_array, dtype=float)

    labels = []
    for label in float_array:
        if 0.1 <= label < 1:
            if colorscale == "pysteps":
                formatting = ",.2f"
            else:
                formatting = ",.1f"
        elif 0.01 <= label < 0.1:
            formatting = ",.2f"
        elif 0.001 <= label < 0.01:
            formatting = ",.3f"
        elif 0.0001 <= label < 0.001:
            formatting = ",.4f"
        elif label >= 1 and label.is_integer():
            formatting = "i"
        else:
            formatting = ",.1f"

        if formatting != "i":
            labels.append(format(label, formatting))
        else:
            labels.append(str(int(label)))

    return labels


def main():
    # 读取8个 nc4 格式的降水数据文件（假设文件名为 file1.nc, file2.nc, ..., file8.nc）
    path = r"D:\Research\p2_gloprec_DL\kw_01_data\data_20200101_20230531\try1_china_TVT\A_TVT_0.6selected_IP\try_multi_prec"
    # 1. 读取并绘制shp文件
    shp_file_path = r'D:\Research\p2_gloprec_DL\kw_01_data\data_20200101_20230531\data_2020-23\china_boundary/bou1_4p.shp'
    china_boundary = gpd.read_file(shp_file_path)
    data_files = []
    for f in os.listdir(path):
        data_files.append(os.path.join(path, f))


    # Create a 4x2 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle('降水图')

    # Loop to plot precipitation data and add China's boundary
    for i, ax in enumerate(axes.flat):
        file_path = data_files[i]
        _plot_precip_from_nc(file_path, ax, china_boundary)

    # Add colorbar on the right side
    redgrey_hex = "#%02x%02x%02x" % (156, 126, 148)
    color_list = [
        redgrey_hex,
        "#640064",
        "#AF00AF",
        "#DC00DC",
        "#3232C8",
        "#0064FF",
        "#009696",
        "#00C832",
        "#64FF00",
        "#96FF00",
        "#C8FF00",
        "#FFFF00",
        "#FFC800",
        "#FFA000",
        "#FF7D00",
        "#E11900",
    ]
    clevs = [
        0.08,
        0.16,
        0.25,
        0.40,
        0.63,
        1,
        1.6,
        2.5,
        4,
        6.3,
        10,
        16,
        25,
        40,
        63,
        100,
        160,
    ]
    clevs_str = _dynamic_formatting_floats(clevs)

    cax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
    cmap = colors.LinearSegmentedColormap.from_list(
                "cmap", color_list, len(clevs) - 1
            )
    norm = colors.BoundaryNorm(clevs, cmap.N)
    cmap.set_bad("gray", alpha=0.5)
    cmap.set_under("none")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array to create the colorbar
    cbar = plt.colorbar(sm, cax=cax, label='Precipitation intensity (mm/h)')
    cbar.ax.set_yticklabels(clevs_str)
    # Adjust layout to make room for the colorbar
    plt.subplots_adjust(right=0.9, hspace=0.05, wspace=-0.5)
    # Display the plot
    plt.show()


if __name__ == '__main__':
    main()