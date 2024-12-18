#!/usr/bin/env python3
# encoding=utf-8

"""
fusion_decision
================

Version: 1.1
Last Modified: 2024-12-18
"""

import rospy
import json
import numpy as np
from std_msgs.msg import String
from yolov5_ros.msg import BoundingBox2D, BoundingBox2DArray
from fusion_utils import read_calib

# 读取标定文件
P0, P1, P2, P3, R0, lidar2camera_matrix, imu2lidar_matrix = read_calib()
intrinsic = P2[:, :3]  # Cam 2(color)
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

# 置信度阈值 
pp_confidence_threshold_fusion = 0.38
yolo_confidence_threshold_fusion = 0.3


class FusionDecision:
    def __init__(self):
        rospy.init_node('fusion_decision', anonymous=False)
        self.pp_results_sub = rospy.Subscriber('pp_results', String, self.pp_results_callback)
        self.yolo_results_sub = rospy.Subscriber('yolo_results', BoundingBox2DArray, self.yolo_results_callback)
        self.fusion_detection_pub = rospy.Publisher('fusion_results', BoundingBox2DArray, queue_size=10)

    def yolo_results_callback(self, yolo_results):
        """
        读取BoundingBox2DArray格式消息,将消息转换后准备送入iou计算
        @param yolo_results:      BoundingBox2DArray
        -------------------------------------------------
        yolov5_ros.msg/BoundingBox2D.msg
        float32 x_min // 左上角x坐标
        float32 y_min // 左上角y坐标
        float32 x_max // 右下角x坐标
        float32 y_max // 右下角y坐标
        float32 value // 置信度
        uint32 label  // 类别
        -------------------------------------------------
        yolov5_ros.msg/BoundingBox2DArray.msg
        Header header // ROS消息头
        BoundingBox2D[] boxes
        """
        self.yolo_detection_bboxes = self.process_yolo_results(yolo_results)

    def pp_results_callback(self, pp_results):
        # if self.cv_image is not None:

        labels_3d, scores_3d, bboxes_3d, box_type_3d = self.load_pp_result(pp_results)

        corners_3d = self.bbox3d_center_to_corners(bboxes_3d)

        projected_corners = self.project_3d_to_2d(corners_3d, intrinsic, extrinsic)

        pp_detected_bboxes = self.bboxes_3d_to_2d(projected_corners, scores_3d, labels_3d)

        # fusion_boxes = self.fusion_decision(self.yolo_detection_bboxes, pp_detected_bboxes)

        all_bboxes = self.yolo_detection_bboxes + pp_detected_bboxes
        final_results = self.weighted_nms(all_bboxes)

        self.publish_fusion_results(final_results)

        print("\n============================")
        print("yolo_detection_bboxes", self.yolo_detection_bboxes)
        print("\n============================")
        print("pp_detected_bboxes", pp_detected_bboxes)
        print("\n============================")
        print("final_results", final_results)
        print("\n============================")

    def process_yolo_results(self, yolo_results):
        # Extract bounding boxes from the message
        bounding_boxes = yolo_results.boxes

        # Prepare a list to store the bounding boxes
        detected_bboxes = []

        # Iterate through each bounding box in the message
        for box in bounding_boxes:
            # Extract the coordinates and other information
            x_min = box.x_min
            y_min = box.y_min
            x_max = box.x_max
            y_max = box.y_max
            confidence = box.value
            label = box.label

            # Create a dictionary to store the bounding box information
            bbox_info = {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'confidence': confidence,
                'label': label, 
                'model': 0
            }

            # Append the bounding box information to the list
            detected_bboxes.append(bbox_info)

        return detected_bboxes

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

    def bbox3d_center_to_corners(self, bboxes):
        """
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
        """
        centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
        # 1. 生成边界框顶点坐标, 按照顺时针方向从最小点排列 （3, 0, 4， 7, 2, 1, 5, 6）   
        bboxes_corners = np.array([[-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0],
                                  [-0.5, 0.5, 1.0], [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0]],
                                  dtype=np.float32)  
        bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]  # [1, 8, 3] * [N, 1, 3] -> [N, 8, 3]
    
        # 2. 绕z轴旋转边界框
        rot_sin, rot_cos = np.sin(angles), np.cos(angles)
        ones, zeros = np.ones_like(rot_cos), np.zeros_like(rot_cos)
        rot_mat = np.array([[rot_cos, rot_sin, zeros],
                            [-rot_sin, rot_cos, zeros],
                            [zeros, zeros, ones]],  # 绕z轴旋转矩阵
                            dtype=np.float32)  # [3, 3, N]

        rot_mat = np.transpose(rot_mat, (2, 0, 1))  # [N, 3, 3]
        bboxes_corners = np.matmul(bboxes_corners, rot_mat)  # [N, 8, 3]

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
        corners_2d_homogeneous = np.matmul(corners_3d_camera[:, :, :3], intrinsic.T)  # (u, v, w)

        # 4. 将齐次坐标的归一化 (去除w分量): [N, 8, 3] -> [N, 8, 2]
        #    (u, v) = (u/w, v/w)
        corners_2d = corners_2d_homogeneous[:, :, :2] / corners_2d_homogeneous[:, :, 2:]  # (u, v)
        return corners_2d

    def bboxes_3d_to_2d(self, projected_2d, scores_3d, labels_3d):
        """
        根据3D边界框八个角点坐标,转换为2D边界框
        @param projected_2d:   np.ndarray  [N, 8, 2]  边界框的8个角点坐标
        @param scores_3d:      np.ndarray  [N]        3D边界框的置信度
        @param labels_3d:      np.ndarray  [N]        3D边界框的类别
        @return:
        """
        # Prepare a list to store the bounding boxes
        pp_detected_bboxes = []

        # 遍历所有的3D边界框，筛选出符合置信度要求的边界框
        for i in range(len(scores_3d)):
            if scores_3d[i] < pp_confidence_threshold_fusion:
                continue
            
            # 获取该边界框的2D投影角点
            corners_2d = projected_2d[i]  # [8, 2]
            
            # 计算x_min, y_min, x_max, y_max
            x_min = np.min(corners_2d[:, 0])
            y_min = np.min(corners_2d[:, 1])
            x_max = np.max(corners_2d[:, 0])
            y_max = np.max(corners_2d[:, 1])
            
            # Store the bounding box information
            bbox_info = {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'confidence': scores_3d[i],
                'label': labels_3d[i], 
                'model': 1
            }

            pp_detected_bboxes.append(bbox_info)

        return pp_detected_bboxes

    @staticmethod
    def iou(bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1 = max(bbox1['x_min'], bbox2['x_min'])
        y1 = max(bbox1['y_min'], bbox2['y_min'])
        x2 = min(bbox1['x_max'], bbox2['x_max'])
        y2 = min(bbox1['y_max'], bbox2['y_max'])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        bbox1_area = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
        bbox2_area = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])

        union_area = bbox1_area + bbox2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0

    def weighted_nms(self, bboxes, iou_threshold=0.5, yolo_weight=0.8, pp_weight=1.0):
        """
        带权重的NMS，优先考虑yolo的结果。

        Args:
            bboxes: 所有边界框的列表，每个边界框是一个字典。
            iou_threshold: IoU阈值。
            yolo_weight: YOLO模型的权重。
            pp_weight: PointPillars模型的权重。

        Returns:
            经过NMS处理后的边界框列表。
        """

        if not bboxes:
            return []

        # 根据置信度降序排列
        bboxes.sort(key=lambda x: x['confidence'], reverse=True)

        final_bboxes = []
        while bboxes:
            best_bbox = bboxes.pop(0)
            final_bboxes.append(best_bbox)

            remaining_bboxes = []
            for bbox in bboxes:
                if best_bbox['label'] != bbox['label']:  # 不同类别不进行iou计算
                    remaining_bboxes.append(bbox)
                    continue

                current_iou = self.iou(best_bbox, bbox)
                if current_iou <= iou_threshold:
                    remaining_bboxes.append(bbox)
                else:
                    # 根据模型权重调整置信度，并保留置信度高的框
                    if best_bbox['model'] == 0 and bbox['model'] == 1:
                        if best_bbox['confidence'] < bbox['confidence'] * pp_weight/yolo_weight:
                            best_bbox = bbox
                    elif best_bbox['model'] == 1 and bbox['model'] == 0:
                        if bbox['confidence'] > best_bbox['confidence'] * pp_weight/yolo_weight:
                            final_bboxes[-1] = bbox
                            best_bbox = bbox

            bboxes = remaining_bboxes

        return final_bboxes

    def publish_fusion_results(self, detections):
        # while not rospy.is_shutdown():
        msg = BoundingBox2DArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_link"

        for detection in detections:
            bbox = BoundingBox2D()
            bbox.x_min = detection['x_min']
            bbox.y_min = detection['y_min']
            bbox.x_max = detection['x_max']
            bbox.y_max = detection['y_max']
            bbox.value = detection['confidence']
            bbox.label = detection['label']
            bbox.model = detection['model']
            msg.boxes.append(bbox)

        self.fusion_detection_pub.publish(msg)

    # def iou(self, box1, box2):
    #     '''
    #     计算两个边界框的交并比（Intersection over Union, IoU）
    #     @param box1: dict 包含x_min, y_min, x_max, y_max
    #     @param box2: dict 包含x_min, y_min, x_max, y_max
    #     @return: float IoU值
    #     '''
    #     x1 = max(box1['x_min'], box2['x_min'])
    #     y1 = max(box1['y_min'], box2['y_min'])
    #     x2 = min(box1['x_max'], box2['x_max'])
    #     y2 = min(box1['y_max'], box2['y_max'])

    #     inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    #     box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    #     box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    #     iou = inter_area / float(box1_area + box2_area - inter_area)

    #     return iou

    # # def non_max_suppression(fusion_boxes, fusion_scores, iou_threshold):
    # #     '''
    # #     非极大值抑制（Non-Maximum Suppression, NMS）
    # #     @param fusion_boxes: 
    # #     @param fusion_scores:       np.ndarray  [N]
    # #     @param iou_threshold:       float IoU阈值
    # #     @return: list 经过NMS处理后的边界框列表
    # #     '''
    # #     if len(fusion_boxes) == 0:
    # #         return []

    # def fusion_decision(self, yolo_bboxes, pp_bboxes):
    #     '''
    #     融合YOLO和PointPillars的检测结果
    #     @param yolo_bboxes: list YOLO检测的边界框
    #     @param pp_bboxes: list PointPillars检测的边界框
    #     @return: list 融合后的边界框
    #     '''
    #     fusion_boxes = []
    #     fusion_scores = []

    #     for yolo_bbox in yolo_bboxes:
    #         for pp_bbox in pp_bboxes:
    #             iou_value = self.iou(yolo_bbox, pp_bbox)
    #             if iou_value > 0.5:  # IoU阈值
    #                 fusion_box = {
    #                     'x_min': min(yolo_bbox['x_min'], pp_bbox['x_min']),
    #                     'y_min': min(yolo_bbox['y_min'], pp_bbox['y_min']),
    #                     'x_max': max(yolo_bbox['x_max'], pp_bbox['x_max']),
    #                     'y_max': max(yolo_bbox['y_max'], pp_bbox['y_max']),
    #                     'confidence': max(yolo_bbox['confidence'], pp_bbox['confidence']),
    #                     'label': yolo_bbox['label']  # Assuming the label is the same
    #                 }
    #                 fusion_boxes.append(fusion_box)
    #                 fusion_scores.append(fusion_box['confidence'])

    #     # Apply Non-Maximum Suppression (NMS)
    #     final_boxes = self.non_max_suppression(fusion_boxes, fusion_scores, iou_threshold=0.3)
    #     return final_boxes

    # def non_max_suppression(self, fusion_boxes, fusion_scores, iou_threshold):
    #     '''
    #     非极大值抑制（Non-Maximum Suppression, NMS）
    #     @param fusion_boxes: list 融合后的边界框
    #     @param fusion_scores: list 融合后的置信度
    #     @param iou_threshold: float IoU阈值
    #     @return: list 经过NMS处理后的边界框列表
    #     '''
    #     if len(fusion_boxes) == 0:
    #         return []

    #     indices = np.argsort(fusion_scores)[::-1]
    #     keep_boxes = []

    #     while len(indices) > 0:
    #         current_index = indices[0]
    #         current_box = fusion_boxes[current_index]
    #         keep_boxes.append(current_box)
    #         indices = indices[1:]

    #         remaining_indices = []
    #         for i in indices:
    #             iou_value = self.iou(current_box, fusion_boxes[i])
    #             if iou_value < iou_threshold:
    #                 remaining_indices.append(i)

    #         indices = remaining_indices

    #     return keep_boxes

# ...existing code...


if __name__ == '__main__':
    try:
        fusion = FusionDecision()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
