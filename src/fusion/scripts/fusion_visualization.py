#!/usr/bin/env python3
# encoding=utf-8

"""
fusion_visualization
================

Version: 2.0
Last Modified: 2024-12-20
"""

import rospy
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import open3d.core as o3c
from sensor_msgs.msg import PointCloud2, Image
from fusion_utils import calibration_param, imgmsg_to_cv2, process_points
from yolov5_ros.msg import BoundingBox2DArray

# 读取标定文件
intrinsic, extrinsic = calibration_param()


class FusionVisualization:
    """
    点云上色渲染, 可视化融合后检测结果
    """
    def __init__(self):
        rospy.init_node('fusion_visualization', anonymous=False)  # 初始化ros节点
        self.pointcloud_sub = rospy.Subscriber('processed_points', PointCloud2, self.pointcloud_callback)  # 订阅点云话题
        self.image_sub = rospy.Subscriber('synced_image', Image, self.image_callback)  # 订阅图像话题
        self.detection_sub = rospy.Subscriber('fusion_results', BoundingBox2DArray, self.detection_callback)  # 订阅融合后结果

        # 初始化变量
        self.depth = None
        self.cv_image = None
        self.initialized = False
        self.height, self.width = None, None

    def image_callback(self, img_msg):
        """
        将Image消息转换为cv2图像
        """
        self.cv_image = imgmsg_to_cv2(img_msg)
        if not self.initialized:
            self.height, self.width = self.cv_image.shape[:2]
            self.initialized = True  # 获取图像height和width, 避免重复初始化

    def pointcloud_callback(self, pcl_msg):
        """
        解析点云消息
        """
        self.points = process_points(pcl_msg)

    def detection_callback(self, msg):
        """
        融合及可视化
        """
        if self.cv_image is not None:
            self.depth = self.points_to_depth(self.points, self.height, self.width, intrinsic, extrinsic)  # 三维点云投影生成深度图
            colored_depth = self.depth_colorize(self.depth)  # 深度图上色渲染
            fusion_image = self.blend_images(self.cv_image, colored_depth)  # 叠加上色后深度图至相机图像
            self.draw_detection(msg, fusion_image)  # 绘制检测结果

    def draw_detection(self, msg, image):
        """
        绘制检测结果
        """
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        thickness = 1       # 标签字体粗细
        font_size = 0.5     # 标签字号
        labels = {
            0: 'Pedestrian',
            1: 'Cyclist',
            2: 'Car'
        }

        for box in msg.boxes:
            x_min = int(box.x_min)
            y_min = int(box.y_min)
            x_max = int(box.x_max)
            y_max = int(box.y_max)
            label = box.label
            confidence = box.value
            model = box.model

            color = (0, 190, 0) if model == 0 else (0, 0, 255)  # YOLO为绿色, PointPillars为红色
            label_text = f"{labels.get(label)}, {confidence:.2f}"  # 获取标签文本

            # 绘制边界框
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # 绘制标签背景
            label_size = cv2.getTextSize(label_text, font, font_size, thickness)[0]
            label_w, label_h = label_size
            label_x = x_min - 1
            label_y = y_min - label_h - 1
            cv2.rectangle(image, (label_x, label_y), (label_x + label_w, label_y + label_h), color, -1)

            # 在框上方绘制标签文本和置信度
            cv2.putText(image, label_text, (x_min, y_min - 2), font, font_size, (255, 255, 255), thickness)

        # 显示图片
        cv2.imshow("Fusion Results", image)
        cv2.waitKey(1)

    @staticmethod
    def points_to_depth(points, height, width, intrinsic, extrinsic):
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
        depth_colored_rgba = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGRA)
        depth_colored_rgba[depth == 0] = [0, 0, 0, 0]
        return depth_colored_rgba

    def blend_images(self, image, colored_depth):
        """
        叠加深度图至相机图像
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
