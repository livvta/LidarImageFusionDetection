#!/usr/bin/env python3
# encoding=utf-8

'''
fusion_visualization
================

Version: 1.2
Last Modified: 2024-11-06 16:35
'''

import rospy
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import open3d.core as o3c
from sensor_msgs.msg import PointCloud2, Image
from fusion_utils import read_calib, imgmsg_to_cv2, process_points


# 读取标定文件
P0, P1, P2, P3, R0, lidar2camera_matrix, imu2lidar_matrix = read_calib( )
intrinsic = P2[:, :3]
extrinsic = np.matmul(R0, lidar2camera_matrix)


class FusionVisualization:
    def __init__(self):
        # 初始化ros节点，节点名称为fusion_visualization
        rospy.init_node('fusion_visualization', anonymous=False)

        # 订阅velodyne_points话题，消息类型为PointCloud2
        self.pointcloud_sub = rospy.Subscriber('processed_points', PointCloud2, self.pointcloud_callback)

        # 订阅image话题，消息类型为Image   
        self.image_sub = rospy.Subscriber('kitti_cam', Image, self.image_callback)


    def image_callback(self, img_msg):
        # 将Image消息转换为cv2
        self.cv_image = imgmsg_to_cv2(img_msg)

        # 获取图像height和width
        if not hasattr(self, 'initialized') or not self.initialized:
            self.height, self.width = self.cv_image.shape[:2]
            self.initialized = True # 避免重复初始化


    def pointcloud_callback(self, pcl_msg):
        #解析点云消息
        points = process_points(pcl_msg)

        if self.cv_image is not None:
            # 三维点云投影生成深度图
            depth = self.points_to_depth(points, self.height, self.width, intrinsic, extrinsic)

            # 深度图上色渲染
            colored_depth = self.depth_colorize(depth)

            # 叠加上色后深度图至相机图像
            fusion_image = self.blend_images(self.cv_image, colored_depth)

            # 绘制检测结果
            self.draw_detection(fusion_image)


    def draw_detection(self, fusion_image):
        # ToDo
        cv2.imshow('Fused Image', fusion_image)
        cv2.waitKey(1)


    def points_to_depth(self, points, height, width, intrinsic, extrinsic):
        """
        三维点云投影生成深度图
        @param points:       np.ndarray  [N, 3]
        @param height:       int
        @param width:        int
        @param intrinsic:    np.ndarray  [3, 3]
        @param extrinsic:    np.ndarray  [4, 4]
        @return:             np.ndarray  [H, W]    float32
        """
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points, dtype=o3c.float32))
        intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
        extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

        o3d_depth_image = pcd.project_to_depth_image(
            width=width,
            height=height,
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            depth_scale=1,
            depth_max=100
        )

        depth_np = np.asarray(o3d_depth_image).squeeze()

        return depth_np


    def depth_colorize(self, depth):
        """
        深度图着色渲染
        @param depth:       np.ndarray  [H, W]
        @return:            np.ndarray  [H, W, 4]    RGBA
        """
        assert depth.ndim == 2
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        cmap = plt.cm.jet
        depth_colored = (255 * cmap(depth)[:, :, :3]).astype(np.uint8)
        depth_colored_rgba =  cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGRA)
        depth_colored_rgba[depth == 0] = [0, 0, 0, 0]

        return depth_colored_rgba


    def blend_images(self, image, colored_depth):
        """
        叠加深度图至相机image
        @param image:               np.ndarray  [H, W, 3]   BGR
        @param colored_depth:       np.ndarray  [H, W, 4]   RGBA
        @return:                    np.ndarray  [H, W, 4]   BGRA
        """
        # 将输入图像转换为BGRA格式
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
         
        # 将 colored_depth 叠加到 image_bgra 上
        blended_image = cv2.addWeighted(image_bgra, 1, colored_depth, 1, 0)

        return blended_image


if __name__ == '__main__':
    try:
        fusion = FusionVisualization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass        
