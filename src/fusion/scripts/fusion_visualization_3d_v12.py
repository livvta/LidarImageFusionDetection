#!/usr/bin/env python3
# encoding=utf-8

'''
fusion_visualization_3d
================

Version: 1.2.1
Last Modified: 2024-12-18

ToDo:
1.传递参数命名不规范
2.project_3d_to_2d函数 未完成
3.bbox3d_center2corners函数 注释未翻译

'''

import rospy
import cv2
import json
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from fusion_utils import read_calib, imgmsg_to_cv2


# 读取标定文件
P0, P1, P2, P3, R0, lidar2camera_matrix, imu2lidar_matrix = read_calib( )
intrinsic = P2[:, :3] # Cam 2(color)
'''
                [fx   0  cx]  
    intrinsic = [ 0  fy  cy]  fx, fy:相机焦距
                [ 0   0   1]  cx, cy:相机主点
'''
extrinsic = np.matmul(R0, lidar2camera_matrix)
'''
    extrinsic = [R | T]       R:旋转矩阵[3, 3]
                              T:平移向量[3, 1] 
'''

# 可视化置信度阈值 
score_threshold=0.3


class FusionVisualization:
    def __init__(self):
        # 初始化ros节点，节点名称为fusion_visualization
        rospy.init_node('fusion_visualization', anonymous=False)

        # 订阅image话题，消息类型为Image   
        self.image_sub = rospy.Subscriber('synced_image', Image, self.image_callback)

        # 接受pp_results话题，消息类型为String
        self.pp_results_sub = rospy.Subscriber('pp_results', String, self.ppdetection_callback)


    def image_callback(self, img_msg):
        # 将Image消息转换为cv2
        self.cv_image = imgmsg_to_cv2(img_msg)

        # 获取图像height和width
        if not hasattr(self, 'initialized') or not self.initialized:
            self.height, self.width = self.cv_image.shape[:2]
            self.initialized = True # 避免重复初始化


    def ppdetection_callback(self, pp_results):
        if self.cv_image is not None:

            labels_3d, scores_3d, bboxes_3d, box_type_3d = self.load_pp_result(pp_results)

            corners_3d = self.bbox3d_center_to_corners(bboxes_3d)

            projected_corners = self.project_3d_to_2d(corners_3d, intrinsic, extrinsic)

            self.draw_3d_bboxes_on_image(self.cv_image, projected_corners, scores_3d)


    def load_pp_result(self, pp_results):
        '''
        加载PointPillars检测结果
        @param pp_results:      std_msgs.msg String
        @return:    tuple    (labels_3d, scores_3d, bboxes_3d, box_type_3d)
                    labels_3d    np.ndarray   [N]
                    scores_3d    np.ndarray   [N]
                    bboxes_3d    np.ndarray   [N, 7]
                    box_type_3d  str    
        '''
        pp_results = json.loads(pp_results.data)
        
        labels_3d_list = pp_results.get("labels_3d", [])
        scores_3d_list = pp_results.get("scores_3d", [])
        bboxes_3d_list = pp_results.get("bboxes_3d", [])
        box_type_3d = pp_results.get("box_type_3d", "")

        labels_3d = np.array(labels_3d_list)
        scores_3d = np.array(scores_3d_list)
        bboxes_3d = np.array(bboxes_3d_list)

        return labels_3d, scores_3d, bboxes_3d, box_type_3d


    def bbox3d_center_to_corners(self, bboxes):
        '''
        通过边界框的位置(底面中点)、尺寸、朝向角参数得到边界框的8个角点坐标
        box_3d: (x, y, z, x_size, y_size, z_size, yaw)
        @param bboxes:      np.ndarray  [N, 7]  
        @return:            np.ndarray  [N, 8, 3]

               ^ z   x            6 ------ 5
               |   /             / |     / |
               |  /             2 -|---- 1 |   
        y      | /              |  |     | | 
        <------|o               | 7 -----| 4
                                |/   o   |/    
                                3 ------ 0 
        x: front, y: left, z: top
        '''
        centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
        # 1. 生成边界框顶点坐标, 按照顺时针方向从最小点排列 （3, 0, 4， 7, 2, 1, 5, 6）   
        bboxes_corners =np.array([[-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0],
                                  [-0.5, 0.5, 1.0],[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0]], 
                                  dtype=np.float32)  
        bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # [1, 8, 3] * [N, 1, 3] -> [N, 8, 3]
    
        # 2. 绕z轴旋转边界框
        rot_sin ,rot_cos = np.sin(angles),np.cos(angles)
        ones ,zeros= np.ones_like(rot_cos), np.zeros_like(rot_cos) 
        rot_mat = np.array([[rot_cos, rot_sin, zeros],
                            [-rot_sin, rot_cos, zeros],
                            [zeros, zeros, ones]],  # 绕z轴旋转矩阵
                            dtype=np.float32) # [3, 3, N]

        rot_mat = np.transpose(rot_mat, (2, 0, 1)) # [N, 3, 3]
        bboxes_corners =  np.matmul(bboxes_corners,rot_mat) # [N, 8, 3]

        # 3. 平移回原中心位置
        bboxes_corners += centers[:, None, :]
    
        return bboxes_corners


    def project_3d_to_2d(self, all_corners, intrinsic, extrinsic):
        """
        将LiDAR坐标系的3D边界框角点投影到画面
        @param all_corners:     np.ndarray  [N, 8, 3]   所有边界框的8个3D角点
        @param intrinsic:       np.ndarray  [3, 3]      相机内参矩阵
        @param extrinsic:       np.ndarray  [3, 4]      相机外参矩阵
        @return:                np.ndarray  [N, 8, 2]   投影后的2D坐标
        """
        num_bbox = all_corners.shape[0]
        
        # 1. 扩展3D角点为齐次坐标: [N, 8, 3] -> [N, 8, 4]
        ones = np.ones((num_bbox, 8, 1))
        all_corners_homogeneous = np.concatenate([all_corners, ones], axis=-1)  # (X, Y, Z, 1)

        # 2. 将LiDAR坐标系下3D角点转换到相机坐标系: [N, 8, 4] -> [N, 8, 4]
        corners_3d_camera = np.matmul(all_corners_homogeneous, extrinsic.T)  # (X', Y', Z', 1)

        # 3. 将相机坐标系中的3D点投影到图像坐标系: [N, 8, 3] -> [N, 8, 3]
        corners_2d_homogeneous = np.matmul(corners_3d_camera[:, :, :3], intrinsic.T) # (u, v, w)

        # 4. 将齐次坐标的归一化 (去除w分量): [N, 8, 3] -> [N, 8, 2]
        #    (u, v) = (u/w, v/w)
        corners_2d = corners_2d_homogeneous[:, :, :2] / corners_2d_homogeneous[:, :, 2:]  # (u, v)
        
        return corners_2d


    def draw_3d_bboxes_on_image(self, img, projected_2d, scores_3d):
        '''
        根据角点坐标绘制3D边界框
        @param img:            np.ndarray  2D图像
        @param projected_2d:   np.ndarray  [N, 8, 2]  边界框的8个角点坐标
        @param scores_3d:      np.ndarray  [N]        3D边界框的置信度
        @return:               np.ndarray  img        绘制了3D边界框的图像
        '''
        img = img.copy()

        # 定义所有的线段连接规则（通过角点索引）
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底部矩形
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶部矩形
            (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直连接线
        ]
        
        # 遍历所有的3D边界框，筛选出符合置信度要求的边界框
        for i in range(len(scores_3d)):
            if scores_3d[i] < score_threshold:
                continue
            
            # 获取该边界框的2D投影角点
            corners_2d = projected_2d[i]  # [8, 2]
            
            # 将投影的2D角点转换为整数坐标
            corners_2d_int = np.round(corners_2d).astype(int)
            
            # 绘制边界框的连接线
            for (start, end) in connections:
                cv2.line(img, tuple(corners_2d_int[start]), tuple(corners_2d_int[end]), (255, 0, 0), 1)
        
        # 显示图像
        cv2.imshow('3D BBoxes on Image', img)
        cv2.waitKey(1)
        
        # 返回绘制了边界框的图像
        return img


if __name__ == '__main__':
    try:
        fusion = FusionVisualization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass        
