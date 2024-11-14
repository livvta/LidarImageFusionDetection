#!/usr/bin/env python3
# encoding=utf-8

'''
pointpillars_node
================

Version: 3.4
Last Modified: 2024-11-09 12:55

更新内容：
10.14：增加Marker消息类型，用于发布检测范围标识
10.15：更换检测方式，将LidarDet3DInferencer改为inference_detector
11.05：callback函数逻辑优化
11.06：增加String消息类型，用于发布原始检测结果; 
11.09：创建processed_points话题，可视化ROI过滤后点云+强度，优化代码

ToDo:
1.过滤后点云可视化检查  done
2.利用內外参，通过相机可视范围过滤点云
2.注释

'''

import rospy
import math
import time
import json
import numpy as np
from mmdet3d.apis import init_model, inference_detector
from mmengine.logging import print_log
from sensor_msgs.msg  import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from std_msgs.msg import String
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
# from mmdet3d.apis import LidarDet3DInferencer

# 模型配置信息
config_file = '/home/harris/model/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class1/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class1.py'
checkpoint_file = '/home/harris/model/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class1/epoch_80.pth'
device = 'cuda:0'

# 初始化模型
model = init_model(config_file, checkpoint_file, device=device)

# 设置检测速度
rate = 2

# 设置是否可视化(BoundingBox + Marker)
rviz_visualization = True

# 设置可视化置信度阈值
confidence_threshold = 0.1

left_angle = math.radians(45)
right_angle = math.radians(45)


class PointPillarsNode:
    def __init__(self):
        # 初始化ros节点，节点名称为pointpillars_node
        rospy.init_node('pointpillars_node', anonymous=False)

        # 订阅velodyne_points话题，消息类型为PointCloud2
        self.pointcloud_sub = rospy.Subscriber('velodyne_points', PointCloud2, self.pointcloud_callback, queue_size=1)

        # 创建processed_points话题，消息类型为PointCloud2，发布ROI处理后的点云，用于rviz可视化
        self.processed_pointcloud_pub = rospy.Publisher('processed_points', PointCloud2, queue_size=10)

        # 创建pp_visualization话题，消息类型为BoundingBoxArray，用于rviz可视化检测结果
        self.pp_visualization_pub = rospy.Publisher('pp_visualization', BoundingBoxArray, queue_size=10)

        # 创建pp_results话题，消息类型为String，用于发布原始检测结果
        self.pp_results_pub = rospy.Publisher('pp_results', String, queue_size=10)

        # 创建angle_boundaries话题，消息类型为Marker，用于可视化检测范围
        self.marker_pub = rospy.Publisher('angle_boundaries', Marker, queue_size=10)        

         # 激活发布检测范围标识Marker
        if rviz_visualization:
            self.publish_angle_boundaries()


    def pointcloud_callback(self, msg):
        try: 
            # 将PointCloud2转换为模型输入格式
            filtered_points = self.convert_pointcloud2(msg)

            # 调用模型检测         
            results = self.pointpillars_detect(filtered_points)

            # 解析原始结果
            bboxes_3d, labels_3d, scores_3d = self.process_results(results)

            # 发布检测结果 String
            self.string_results_pub(bboxes_3d, labels_3d, scores_3d)

            if rviz_visualization:
                # 发布ROI处理后的点云 PointCloud2
                self.publish_processed_pointcloud(filtered_points)

                # 发布3D边界框 BoundingBoxArray      
                self.bboxes_array_pub(bboxes_3d, labels_3d, scores_3d)
            
            time.sleep(1 / self.rate)            

        except Exception as e:
            rospy.logerr(f"PointPillarsNode error: {e}")
            return


    def convert_pointcloud2(self, pcl_msg):
        """ 将PointCloud2点云数据转换为模型输入格式，并进行ROI过滤 """
        pcl_data = pc2.read_points(pcl_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        pcl_array = np.array(list(pcl_data), dtype=np.float32)

        x = pcl_array[:, 0]
        y = pcl_array[:, 1]
        z = pcl_array[:, 2]
        i = pcl_array[:, 3] if pcl_array.shape[1] == 4 else np.zeros((pcl_array.shape[0],))

        # # 定义角度限制ROI
        # left_angle = math.radians(45)  # 左45度
        # right_angle = math.radians(45)  # 右45度

        # # Calculate tangent values
        # left_tan = np.tan(left_angle)
        # right_tan = np.tan(right_angle)

        # 直接使用常量(45度)
        left_tan = 1
        right_tan = 1

        # 提前x坐标
        x_offset = x - 1

        # 过滤点：只保留前方并在角度限制内的点
        mask = (x > 1) & (y >= -x_offset * right_tan) & (y <= x_offset * left_tan)

        # 应用掩膜过滤点并返回
        points = np.column_stack((x[mask], y[mask], z[mask], i[mask]))

        return points


    def create_pointcloud2_msg(self, points):
        """ 从处理后的点云数据创建PointCloud2消息 """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "velodyne"

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]

        point_cloud_msg = pc2.create_cloud(header, fields, points)
        
        return point_cloud_msg


    def publish_processed_pointcloud(self, processed_points):
        """
        发布处理后的点云
        """
        point_cloud_msg = self.create_pointcloud2_msg(processed_points)
        
        # 发布消息
        self.processed_pointcloud_pub.publish(point_cloud_msg)
        rospy.loginfo("Processed point cloud published.")


    def publish_angle_boundaries(self):
        rate = rospy.Rate(1)  # 1 Hz

        # 只创建一次 Marker
        marker = Marker()
        marker.header.frame_id = "velodyne"
        marker.ns = "angle_boundaries"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # 计算边界线的端点
        line_length = 10
        x_origin = 1.0  # 原点在 x=1
        left_angle = math.radians(45)
        right_angle = math.radians(45)

        # 计算左边界和右边界的端点
        left_boundary = (x_origin + line_length, line_length * math.tan(left_angle))  # x + d
        right_boundary = (x_origin + line_length, -line_length * math.tan(right_angle))  # x + d  

        # 设置线条宽度
        marker.scale.x = 0.1

        # 设置线条颜色rgb&a
        marker.color.r = 1.0  # 红色
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # 不透明              

        while not rospy.is_shutdown():

            marker.header.stamp = rospy.Time.now()
            marker.points.clear()  # 清除旧的点

            # 更新 Marker points
            marker.points.append(Point(x_origin, 0, 0))  # 起点
            marker.points.append(Point(left_boundary[0], left_boundary[1], 0))  # 左边界
            marker.points.append(Point(x_origin, 0, 0))  # 起点
            marker.points.append(Point(right_boundary[0], right_boundary[1], 0))  # 右边界

            # 发布 Marker
            self.marker_pub.publish(marker)
            rate.sleep()


    def pointpillars_detect(self, pcd):
        '''
        PointPillars模型推理
        @param pcd      np.ndarray  [N, 4]
        @return:        Det3DDataSample
        '''
        results, _ = inference_detector(model, pcd)
        # print(type(results))
        # print(results)

        return results


    def process_results(self, results_mmdet3d):
        '''解析原始结果'''
        bboxes_3d = results_mmdet3d.pred_instances_3d.bboxes_3d.cpu().numpy()
        labels_3d = results_mmdet3d.pred_instances_3d.labels_3d.cpu().numpy()
        scores_3d = results_mmdet3d.pred_instances_3d.scores_3d.cpu().numpy()

        return bboxes_3d, labels_3d, scores_3d


    def string_results_pub(self, bboxes_3d, labels_3d, scores_3d):
        '''
        发布String类型检测消息
        @param bboxes_3d:       np.ndarray  [N, 7]
        @param labels_3d:       np.ndarray  [N]
        @param scores_3d:       np.ndarray  [N]
        '''
        results_dict = {
            "labels_3d": labels_3d.tolist(),
            "scores_3d": scores_3d.tolist(),
            "bboxes_3d": bboxes_3d.tolist(), 
            "box_type_3d": "LiDAR"
        }

        results_json = json.dumps(results_dict)  # dict2json
        self.pp_results_pub.publish(results_json)


    def bboxes_array_pub(self, bboxes_3d, labels_3d, scores_3d):
        '''
        jsk_recognition_msgs/BoundingBox.msg
        Header header //消息头
        geometry_msgs/Pose pose	//位姿
        geometry_msgs/Vector3 dimensions //尺寸
        float32 value //置信度
        unit32 label //标签
        '''

        bbox_array = BoundingBoxArray()
        current_time = rospy.Time.now()  # 只计算一次
        bbox_array.header.stamp = current_time
        bbox_array.header.frame_id = "velodyne"

        # 过滤符合阈值的边界框
        valid_indices = scores_3d >= confidence_threshold
        filtered_bboxes_3d = bboxes_3d[valid_indices]
        filtered_labels_3d = labels_3d[valid_indices]
        filtered_scores_3d = scores_3d[valid_indices]

        boxes = []
        for i in range(len(filtered_labels_3d)):
            bbox = BoundingBox()
            bbox.header.stamp = current_time
            bbox.header.frame_id = "velodyne"

            # 解析边界框参数
            x, y, z, length, width, height, rotation = filtered_bboxes_3d[i]

            bbox.pose.position.x = x
            bbox.pose.position.y = y
            bbox.pose.position.z = z / 2

            bbox.dimensions.x = length
            bbox.dimensions.y = width
            bbox.dimensions.z = height

            # 计算四元数
            qx = 0.0
            qy = 0.0
            qz = math.sin(rotation / 2)
            qw = math.cos(rotation / 2)
            bbox.pose.orientation.x = qx
            bbox.pose.orientation.y = qy
            bbox.pose.orientation.z = qz
            bbox.pose.orientation.w = qw

            bbox.value = filtered_scores_3d[i]
            bbox.label = filtered_labels_3d[i]

            # 添加到边界框列表
            boxes.append(bbox)

        bbox_array.boxes.extend(boxes)
        self.pp_visualization_pub.publish(bbox_array)


if __name__ == '__main__':
    try:
        node = PointPillarsNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

