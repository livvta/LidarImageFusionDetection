#!/usr/bin/env python
# encoding=utf-8

"""
pointcloud_visualization.py
===========================
Version: 1.0
Last Modified: 2024-11-18 01:00
"""

import open3d as o3d
import numpy as np
from preprocessing_utils import read_bin_file

def visualize_pointcloud(point_cloud):
    """
    @param point_cloud: np.ndarray  [N, 4]  点云数据
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud Visualization')
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # bin_file_path = 'c:/Users/livvta/OneDrive/EDU/graduation_project/detection_ws/src/pointpillars_ros/scripts/downsampling/data/000000.bin'
    bin_file_path = 'c:/Users/livvta/OneDrive/EDU/graduation_project/detection_ws/src/pointpillars_ros/scripts/downsampling/data_output/000001.bin'
    point_cloud = read_bin_file(bin_file_path)
    visualize_pointcloud(point_cloud)
