#!/usr/bin/env python3
# encoding=utf-8

"""
bbox3d_visualization
================

Version: 1.2.1
Last Modified: 2024-12-18
"""

import json
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from fusion_utils import calibration_param, imgmsg_to_cv2, bbox3d_center_to_corners, project_3d_to_2d

# 读取标定文件
intrinsic, extrinsic = calibration_param()

# 可视化置信度阈值
score_threshold = 0.3


class Bbox3DVisualization:
    """
    将Lidar坐标系的3D边界框绘制在Image上并可视化
    """
    def __init__(self):
        rospy.init_node('fusion_visualization', anonymous=False)
        self.image_sub = rospy.Subscriber('synced_image', Image, self.image_callback)
        self.pp_results_sub = rospy.Subscriber('pp_results', String, self.ppdetection_callback)
        self.cv_image = None  # 初始化cv_image为None
        self.initialized = False  # 初始化initialized为False

    def image_callback(self, img_msg):
        self.cv_image = imgmsg_to_cv2(img_msg)  # 将Image消息转换为cv2

        # # 获取图像height和width, 避免重复初始化
        # if not self.initialized:
        #     self.height, self.width = self.cv_image.shape[:2]
        #     self.initialized = True

    def ppdetection_callback(self, pp_results):
        if self.cv_image is not None:
            _, scores_3d, bboxes_3d, _ = self.load_pp_result(pp_results)
            corners_3d = bbox3d_center_to_corners(bboxes_3d)
            projected_corners = project_3d_to_2d(corners_3d, intrinsic, extrinsic)
            self.draw_3d_bboxes_on_image(self.cv_image, projected_corners, scores_3d)

    def load_pp_result(self, pp_results):
        """
        加载PointPillars检测结果
        @param pp_results:      std_msgs.msg String
        @return:    tuple    (labels_3d, scores_3d, bboxes_3d, box_type_3d)
                    labels_3d    np.ndarray   [N]
                    scores_3d    np.ndarray   [N]
                    bboxes_3d    np.ndarray   [N, 7]
                    box_type_3d  str
        """
        pp_results = json.loads(pp_results.data)

        labels_3d_list = pp_results.get("labels_3d", [])
        scores_3d_list = pp_results.get("scores_3d", [])
        bboxes_3d_list = pp_results.get("bboxes_3d", [])
        box_type_3d = pp_results.get("box_type_3d", "")

        labels_3d = np.array(labels_3d_list)
        scores_3d = np.array(scores_3d_list)
        bboxes_3d = np.array(bboxes_3d_list)
        return labels_3d, scores_3d, bboxes_3d, box_type_3d

    def draw_3d_bboxes_on_image(self, img, projected_2d, scores_3d):
        """
        根据角点坐标绘制3D边界框
        @param img:            np.ndarray  2D图像
        @param projected_2d:   np.ndarray  [N, 8, 2]  边界框的8个角点坐标
        @param scores_3d:      np.ndarray  [N]        3D边界框的置信度
        @return:               np.ndarray  img        绘制了3D边界框的图像
        """
        img_copy = img.copy()

        # 定义所有的线段连接规则（通过角点索引）
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底部矩形
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶部矩形
            (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直连接线
        ]

        # 遍历所有的3D边界框, 筛选出符合置信度要求的边界框
        for i, score in enumerate(scores_3d):
            if score < score_threshold:
                continue

            # 获取该边界框的2D投影角点
            corners_2d = projected_2d[i]  # [8, 2]

            # 将投影的2D角点转换为整数坐标
            corners_2d_int = np.round(corners_2d).astype(int)

            # 绘制边界框的连接线
            for (start, end) in connections:
                cv2.line(img_copy, tuple(corners_2d_int[start]),
                         tuple(corners_2d_int[end]), (255, 0, 0), 1)

        # 显示图像
        cv2.imshow('3D BBoxes on Image', img_copy)
        cv2.waitKey(1)

        # 返回绘制了边界框的图像
        return img_copy


if __name__ == '__main__':
    try:
        fusion = Bbox3DVisualization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
