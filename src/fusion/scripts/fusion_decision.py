#!/usr/bin/env python3
# encoding=utf-8

"""
fusion_decision
================

Version: 2.1
Last Modified: 2024-12-20
"""

import json
import argparse
import rospy
import numpy as np
from std_msgs.msg import String
from fusion_utils import calibration_param
from yolov5_ros.msg import BoundingBox2D, BoundingBox2DArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

# 读取标定文件
intrinsic, extrinsic = calibration_param()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fusion Decision System")
    parser.add_argument('-pp_conf', type=float, default=0.38, help='Confidence threshold for PP results')
    parser.add_argument('-yolo_conf', type=float, default=0.3, help='Confidence threshold for YOLO results')
    parser.add_argument('-iou_thresh', type=float, default=0.5, help='IoU threshold for NMS')
    parser.add_argument('-yolo_weight', type=float, default=0.8, help='Weight for YOLO results')
    parser.add_argument('-pp_weight', type=float, default=1.0, help='Weight for PointPillars results')
    args = parser.parse_args()
    return args


class FusionDecision:
    """
    3D & 2D 边界框结果融合决策系统。
    接收边界框数据, 进行处理并融合结果。
    """
    def __init__(self, yolo_conf, pp_conf, iou_threshold, yolo_weight, pp_weight):
        rospy.init_node('fusion_decision', anonymous=False)

        # 初始化参数
        self.yolo_detection_bboxes = []
        self.pp_confidence_threshold = pp_conf
        self.yolo_confidence_threshold = yolo_conf
        self.iou_threshold = iou_threshold
        self.yolo_weight = yolo_weight
        self.pp_weight = pp_weight

        # 创建订阅者
        self.pp_results_sub = Subscriber('pp_results', String)
        self.yolo_results_sub = Subscriber('yolo_results', BoundingBox2DArray)

        # 同步订阅者
        self.sync = ApproximateTimeSynchronizer(
            [self.pp_results_sub, self.yolo_results_sub],
            queue_size=10,
            slop=0.1  # 时间戳误差允许范围（秒）
        )
        self.sync.registerCallback(self.fusion_callback)

        # 发布融合结果
        self.fusion_detection_pub = rospy.Publisher('fusion_results', BoundingBox2DArray, queue_size=10)

    def fusion_callback(self, pp_results, yolo_results):
        """
        同步接收并融合来自 PointPillars 和 YOLO 的结果。
        """
        # 处理 PointPillars 结果
        labels_3d, scores_3d, bboxes_3d, _ = self.load_pp_result(pp_results)
        corners_3d = self.bbox3d_center_to_corners(bboxes_3d)
        projected_corners = self.project_3d_to_2d(corners_3d, intrinsic, extrinsic)
        pp_detected_bboxes = self.bboxes_3d_to_2d(projected_corners, scores_3d, labels_3d)

        # 处理 YOLO 结果
        yolo_detected_bboxes = self.process_yolo_results(yolo_results)

        # 融合结果
        final_results = self.weighted_nms(pp_detected_bboxes + yolo_detected_bboxes)

        # 发布融合后的结果
        self.publish_fusion_results(final_results)

    # def pp_results_callback(self, pp_results):
    #     labels_3d, scores_3d, bboxes_3d, _ = self.load_pp_result(pp_results)
    #     corners_3d = self.bbox3d_center_to_corners(bboxes_3d)
    #     projected_corners = self.project_3d_to_2d(corners_3d, intrinsic, extrinsic)
    #     pp_detected_bboxes = self.bboxes_3d_to_2d(projected_corners, scores_3d, labels_3d)
    #     final_results = self.weighted_nms(self.yolo_detection_bboxes + pp_detected_bboxes)
    #     self.publish_fusion_results(final_results)

    #     # print("\n============================")
    #     # print("yolo_detection_bboxes", self.yolo_detection_bboxes)
    #     # print("\n============================")
    #     # print("pp_detected_bboxes", pp_detected_bboxes)
    #     # print("\n============================")
    #     # print("final_results", final_results)
    #     # print("\n============================")

    # def yolo_results_callback(self, yolo_results):
    #     """
    #     读取BoundingBox2DArray格式消息,将消息转换后准备送入iou计算
    #     @param yolo_results:      BoundingBox2DArray
    #     """
    #     self.yolo_detection_bboxes = self.process_yolo_results(yolo_results)

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
        # 1. 计算边界框顶点坐标, 按照顺时针方向从最小点排列 （3, 0, 4, 7, 2, 1, 5, 6）
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
        根据2D平面3D边界框的八个角点坐标,转换为2D边界框
        @param projected_2d:   np.ndarray  [N, 8, 2]  边界框的8个角点坐标
        @param scores_3d:      np.ndarray  [N]        3D边界框的置信度
        @param labels_3d:      np.ndarray  [N]        3D边界框的类别
        @return:
        """
        pp_detected_bboxes = []
        for i in range(len(scores_3d)):  # 置信度筛选
            if scores_3d[i] < self.pp_confidence_threshold:
                continue

            # 获取该边界框的2D投影角点
            corners_2d = projected_2d[i]  # [8, 2]

            # 计算x_min, y_min, x_max, y_max
            x_min = np.min(corners_2d[:, 0])
            y_min = np.min(corners_2d[:, 1])
            x_max = np.max(corners_2d[:, 0])
            y_max = np.max(corners_2d[:, 1])

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

    def process_yolo_results(self, yolo_results):
        bounding_boxes = yolo_results.boxes
        yolo_detected_bboxes = []

        for box in bounding_boxes:
            x_min = box.x_min
            y_min = box.y_min
            x_max = box.x_max
            y_max = box.y_max
            confidence = box.value
            label = box.label

            if confidence < self.yolo_confidence_threshold:
                continue

            bbox_info = {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'confidence': confidence,
                'label': label,
                'model': 0
            }
            yolo_detected_bboxes.append(bbox_info)

        return yolo_detected_bboxes

    @staticmethod
    def iou(bbox1, bbox2):
        """
        计算两个边界框的IoU (Intersection over Union)
        @param bbox1:      dict      边界框1, 包含 'x_min', 'y_min', 'x_max', 'y_max'
        @param bbox2:      dict      边界框2, 包含 'x_min', 'y_min', 'x_max', 'y_max'
        @return:           float     计算得到的IoU值, 若没有交集则返回0
        """
        x1 = max(bbox1['x_min'], bbox2['x_min'])
        y1 = max(bbox1['y_min'], bbox2['y_min'])
        x2 = min(bbox1['x_max'], bbox2['x_max'])
        y2 = min(bbox1['y_max'], bbox2['y_max'])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        bbox1_area = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
        bbox2_area = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])

        union_area = bbox1_area + bbox2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0

    def weighted_nms(self, bboxes):
        """
        带权重的NMS (Non-Maximum Suppression)
        @param bboxes:     list      边界框列表, 每个边界框为字典
        @return:           list      经过NMS处理后的边界框列表
        """
        if not bboxes:
            return []

        # 根据置信度降序排列
        bboxes.sort(key=lambda x: x['confidence'], reverse=True)

        final_bboxes = []
        bboxes = [bbox.copy() for bbox in bboxes]  # 避免修改原始输入数据

        while bboxes:
            # 取出当前置信度最高的框
            best_bbox = bboxes.pop(0)
            final_bboxes.append(best_bbox)

            def should_keep(bbox):
                """判断是否保留当前bbox"""
                if best_bbox['label'] != bbox['label']:
                    return True  # 不同类别保留

                current_iou = self.iou(best_bbox, bbox)
                if current_iou <= self.iou_threshold:
                    return True  # IoU低于阈值保留

                # 根据模型权重调整置信度比较
                if best_bbox['model'] == 0 and bbox['model'] == 1:
                    if best_bbox['confidence'] < bbox['confidence'] * self.pp_weight / self.yolo_weight:
                        best_bbox.update(bbox)
                elif best_bbox['model'] == 1 and bbox['model'] == 0:
                    if bbox['confidence'] > best_bbox['confidence'] * self.pp_weight / self.yolo_weight:
                        best_bbox.update(bbox)
                return False

            # 更新bboxes列表, 仅保留需要的框
            bboxes = list(filter(should_keep, bboxes))

        return final_bboxes

    def publish_fusion_results(self, detections):
        """
        发布融合后的检测结果
        @param detections:  list      检测结果列表
        @return:            None      无返回值, 直接发布消息
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


if __name__ == '__main__':
    try:
        args = parse_arguments()
        print(args)
        fusion = FusionDecision(args.yolo_conf, args.pp_conf,  # 置信度阈值
                                args.iou_thresh,  # iou 阈值
                                args.yolo_weight, args.pp_weight)  # nms融合权重
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
