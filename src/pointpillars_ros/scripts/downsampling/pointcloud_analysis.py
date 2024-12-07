#!/usr/bin/env python
# encoding=utf-8

'''
pointcloud_analysis.py
=================
Version: 1.0
Last Modified: 2024-11-18 01:11

可视化分析点云数据的方位角和高度角
'''

import numpy as np
import matplotlib.pyplot as plt
from preprocessing_utils import read_bin_file

def analyze_pointcloud_data(data):
    '''
    @param data: np.ndarray  [N, 4]  点云数据
    '''
    hori_angle = np.degrees(np.arctan2(data[:,0], data[:,1])) # 计算方向角
    elev_angle = np.degrees(np.arctan2(data[:,2], np.sqrt(data[:,0]**2 + data[:,1]**2))) # 计算俯仰角

    plt.rcParams['figure.dpi'] = 130  # 设置分辨率
    plt.figure()
    plt.subplot(2, 1, 1)
    # plt.plot(hori_angle, label='Line')  # 连线
    plt.scatter(range(len(hori_angle)), hori_angle, s=0.001, c='blue', label='Points')  # 画点
    plt.xlabel('Point Index')
    plt.ylabel('Azimuthal Angle (Degrees)')
    plt.title('Azimuthal Angle Analysis')
    plt.ylim(-220, 220)
    plt.yticks(np.arange(-180, 181, 90))
    # plt.legend()  # 添加图例

    plt.subplot(2, 1, 2)
    # plt.plot(elev_angle, label='Line')
    plt.scatter(range(len(elev_angle)), elev_angle, s=0.001, c='blue', label='Points')
    plt.xlabel('Point Index')
    plt.ylabel('Elevation Angle (Degrees)')
    plt.title('Elevation Angle Analysis')
    # plt.legend()  # 添加图例
    plt.ylim(-30, 6)
    plt.yticks(np.arange(-24.8, 3, 6.7))
    # HDL-64E中，最大的水平角是2°，最小的水平角是-24.8°，中间的水平角均匀变化，相邻的两个发射器的水平角相差大概0.4°

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_paths = [
        # '/home/harris/detection_ws/src/pointpillars_ros/scripts/downsampling/data_output/000000.bin'
        r'C:\Users\livvta\OneDrive\EDU\graduation_project\detection_ws\src\pointpillars_ros\scripts\downsampling\data_output\000000.bin'
    ]
    for file_path in file_paths:
        point_cloud = read_bin_file(file_path)
        analyze_pointcloud_data(point_cloud)

