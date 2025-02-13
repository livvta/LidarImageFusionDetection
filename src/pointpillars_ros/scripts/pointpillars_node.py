#!/usr/bin/env python3
# encoding=utf-8

"""
pointpillars_node
================

Version: 3.5
Last Modified: 2024-11-14 20:10

更新内容:
10.14: 增加Marker消息类型, 用于发布检测范围标识
10.15: 更换检测方式, 将LidarDet3DInferencer改为inference_detector
11.05: callback函数逻辑优化
11.06: 增加String消息类型, 用于发布原始检测结果;
11.09: 创建processed_points话题, 可视化ROI过滤后点云+强度, 优化代码
11.14: 将点云过滤及Marker发布逻辑移至lidar_cam_pub.py
"""

import rospy
import math
import json
import numpy as np
from mmdet3d.apis import init_model, inference_detector
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

# 基于KITTI训练的模型, batch_size = 6
config_file = '/home/harris/model/mmdetection3d/work_dirs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/config.py'
checkpoint_file = '/home/harris/model/mmdetection3d/work_dirs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.pth'
device = 'cuda:0'

# 基于KITTI训练的模型, batch_size = 4
# config_file = '/home/harris/model/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class1/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class1.py'
# checkpoint_file = '/home/harris/model/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class1/epoch_80.pth'

# 基于KITTI_16训练的模型
# config_file = '/home/harris/model/mmdetection3d/work_dirs/pointpillars_my_config_hv_secfpn/pointpillars_my_config_hv_secfpn.py'
# checkpoint_file = '/home/harris/model/mmdetection3d/work_dirs/pointpillars_my_config_hv_secfpn/epoch_80.pth'

# 是否启用可视化功能
rviz_visualization = True

# 设置置信度阈值（仅对可视化有效)
pp_confidence_threshold = 0.3


class PointPillarsNode:
    def __init__(self):
        self.pointpillars_model = init_model(config_file, checkpoint_file, device=device)  # 初始化模型
        rospy.init_node('pointpillars_node', anonymous=False)  # 初始化pointpillars_node节点
        self.pointcloud_sub = rospy.Subscriber('processed_points', PointCloud2, self.pointcloud_callback, queue_size=1)  # 订阅点云话题
        self.pp_results_pub = rospy.Publisher('pp_results', String, queue_size=10)  # 发布原始检测结果
        self.pp_visualization_pub = rospy.Publisher('pp_visualization', BoundingBoxArray, queue_size=10)  # 用于rviz可视化检测结果


    def pointcloud_callback(self, msg):
        try:
            # 将PointCloud2转换为模型输入格式
            filtered_points = self.convert_pointcloud2(msg)

            # 调用PointPillars模型检测
            results = self.pointpillars_detect(filtered_points)

            # 解析原始结果
            labels_3d, scores_3d, bboxes_3d = self.process_results(results)

            # 发布检测结果 String
            self.publish_string_results(labels_3d, scores_3d, bboxes_3d)

            # 发布3D边界框 BoundingBoxArray
            if rviz_visualization:
                self.publish_bboxes_array(labels_3d, scores_3d, bboxes_3d)

        except Exception as e:
            rospy.logerr(f"PointPillarsNode error: {e}")
            return


    def convert_pointcloud2(self, pcl_msg):
        """
        将PointCloud2点云数据转换为模型输入格式, 并进行ROI过滤
        @param pcl_msg:     PointCloud2
        @return:            np.ndarray  [N, 4]
        """
        pcl_data = pc2.read_points(pcl_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        pcl_array = np.array(list(pcl_data), dtype=np.float32)
        return pcl_array


    def pointpillars_detect(self, pcd):
        """
        使用PointPillars模型进行点云检测
        @param pcd:     np.ndarray  [N, 4]
        @return:        Det3DDataSample    原始检测结果
        """
        results, _ = inference_detector(self.pointpillars_model, pcd)
        return results


    def process_results(self, results_mmdet3d):
        """
        处理检测结果, 提取3D边界框、标签和置信度得分
        @param results_mmdet3d:     Det3DDataSample
        @return:    tuple    (labels_3d, scores_3d, bboxes_3d, box_type_3d)
                    labels_3d    np.ndarray   [N]
                    scores_3d    np.ndarray   [N]
                    bboxes_3d    np.ndarray   [N, 7]
        """
        labels_3d = results_mmdet3d.pred_instances_3d.labels_3d.cpu().numpy()
        scores_3d = results_mmdet3d.pred_instances_3d.scores_3d.cpu().numpy()
        bboxes_3d = results_mmdet3d.pred_instances_3d.bboxes_3d.cpu().numpy()
        return labels_3d, scores_3d, bboxes_3d


    def publish_string_results(self, labels_3d, scores_3d, bboxes_3d):
        """
        将检测结果以String消息类型发布
        @param labels_3d:       np.ndarray  [N]
        @param scores_3d:       np.ndarray  [N]
        @param bboxes_3d:       np.ndarray  [N, 7]
        """
        results_dict = {
            "labels_3d": labels_3d.tolist(),
            "scores_3d": scores_3d.tolist(),
            "bboxes_3d": bboxes_3d.tolist(),
            "box_type_3d": "LiDAR"
        }

        results_json = json.dumps(results_dict)  # dict2json
        self.pp_results_pub.publish(results_json)


    def publish_bboxes_array(self, labels_3d, scores_3d, bboxes_3d):
        """
        将检测结果以BoundingBoxArray消息类型发布, 用于rviz可视化
        @param bboxes_3d:       np.ndarray  [N, 7]
        @param labels_3d:       np.ndarray  [N]
        @param scores_3d:       np.ndarray  [N]
        -------------------------------------------------
        jsk_recognition_msgs/BoundingBox.msg
        Header header // 消息头
        geometry_msgs/Pose pose	// 位姿
        geometry_msgs/Vector3 dimensions // 尺寸
        float32 value // 置信度
        unit32 label  // 标签
        """
        bbox_array = BoundingBoxArray()
        current_time = rospy.Time.now()  # 只计算一次
        bbox_array.header.stamp = current_time
        bbox_array.header.frame_id = "velodyne"

        # 过滤符合阈值的边界框
        valid_indices = scores_3d >= pp_confidence_threshold
        filtered_bboxes_3d = bboxes_3d[valid_indices]
        filtered_labels_3d = labels_3d[valid_indices]
        filtered_scores_3d = scores_3d[valid_indices]

        for i in range(len(filtered_labels_3d)):
            # 解析边界框参数
            x, y, z, length, width, height, rotation = filtered_bboxes_3d[i]
            qx = 0.0
            qy = 0.0
            qz = math.sin(rotation / 2)
            qw = math.cos(rotation / 2) # 计算四元数

            bbox = BoundingBox()
            bbox.header.stamp = current_time
            bbox.header.frame_id = "velodyne"
            bbox.pose.position.x = x
            bbox.pose.position.y = y
            bbox.pose.position.z = z / 2
            bbox.dimensions.x = length
            bbox.dimensions.y = width
            bbox.dimensions.z = height
            bbox.pose.orientation.x = qx
            bbox.pose.orientation.y = qy
            bbox.pose.orientation.z = qz
            bbox.pose.orientation.w = qw
            bbox.value = filtered_scores_3d[i]
            bbox.label = filtered_labels_3d[i]
            bbox_array.boxes.append(bbox)

        self.pp_visualization_pub.publish(bbox_array)


if __name__ == '__main__':
    try:
        node = PointPillarsNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
