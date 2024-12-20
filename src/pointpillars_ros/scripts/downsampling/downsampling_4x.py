#!/usr/bin/env python
# encoding=utf-8

"""
downsampling_4x.py
==================
Version: 1.1
Last Modified: 2024-11-26 16:46

功能：
- 根据激光雷达的线束特性, 对点云数据进行4倍下采样
"""

import numpy as np
import os
from preprocessing_utils import read_bin_file, write_bin_file


def find_beam_ends(points):
    """
    判断线数及每条激光线的最后一个点
    @param points: np.array [N, 4] 原始的点云数据
    """
    cloud_size = len(points)  # 点云数量
    beam_end_indices = []  # 每条线的最后一个点的索引
    beam_count = 0    # 线数
    prev_angle = None  # 上一个点的方位角
    prev2_angle = None  # 上上个点的方位角
    is_discarded = True # 是否舍弃当前点
    discard_count = 0 # 初始化舍弃计数

    for i in range(cloud_size):
        # 计算当前点的方位角
        angle = np.arctan2(points[i, 1], -points[i, 0]) * 180 / np.pi

        # 初始化
        if i == 0:
            prev_angle = angle
            prev2_angle = angle

        # 前十个点不进入下一线束的判断, 避免噪声影响
        if is_discarded and discard_count < 10:
            discard_count += 1
        else:
            is_discarded = False

        # 判断是否属于下一条线
        if (angle - prev_angle) >= 90 and not is_discarded and (angle - prev2_angle) >= 90:
            beam_count += 1  # 线数加1
            beam_end_indices.append(i)  # 记录当前线的最后一个点的索引

            is_discarded = True  # 舍弃当前点
            discard_count = 0  # 重置舍弃计数

        # 更新上一个点和上上个点的方位角
        prev2_angle = prev_angle
        prev_angle = angle

    beam_end_indices.append(cloud_size - 1)  # 将最后一个点的索引加入

    if beam_count != 63:
        print("Warimg: beams not equal to 64.  beams:", beam_count)
    return beam_end_indices

def downsample_pointcloud(points, beam_end_indices):
    """
    根据分线的index结果, 按四条线选择一条, 生成新点云
    @param points: np.array [N, 4] 原始的点云数据
    @param beam_end_indices: list 每条线的最后一个点的索引
    @return: np.array [M, 4] 下采样后的点云数据
    """
    selected_points = []
    for i in range(0, len(beam_end_indices) - 1, 4):
        start_idx = beam_end_indices[i]
        end_idx = beam_end_indices[i + 1]
        selected_points.extend(points[start_idx:end_idx])
    return np.array(selected_points)

def process_folder(input_folder, output_folder):
    """
    处理文件夹下的所有bin文件
    @param input_folder: 输入文件夹
    @param output_folder: 输出文件夹
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_names = sorted(os.listdir(input_folder))  # 按文件名称排序
    for file_name in file_names:
        if file_name.endswith('.bin'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            points = read_bin_file(input_file_path)
            beam_end_indices = find_beam_ends(points)
            selected_points = downsample_pointcloud(points, beam_end_indices)
            write_bin_file(output_file_path, selected_points)
            print(f"Processed {file_name}")


if __name__ == "__main__":
    input_folder = '/media/harris/PM981/data_object_velodyne/testing/velodyne'
    output_folder = '/media/harris/PM981/data_object_velodyne/testing/velodyne_16'
    process_folder(input_folder, output_folder)