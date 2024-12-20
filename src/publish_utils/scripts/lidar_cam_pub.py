#!/usr/bin/env python2
# coding:utf-8

"""
lidar_cam_pub
================

Version: 2.0
Last Modified: 2024-11-28 21:04

由于受计算平台检测性能限制, 无法做到信息实时检测(rate = 10)
因此, 本程序目的在于同步激光雷达点云与图像消息, 并对发布频率进行控制。

此程序功能：
1. 接收来自Velodyne VLP-16激光雷达点云消息, 过滤掉无效点云, 通过ros发布PointCloud2消息。
"""

import math
import argparse
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pc2

PUB_IMAGE = True

def parse_arguments():
    parser = argparse.ArgumentParser(description='Synchronize and publish lidar and camera data with controlled frequency.')
    parser.add_argument('-r', type=int, default=5, help='Divisor for the publish rate. Default is 5 (10Hz -> 2Hz).')
    args = parser.parse_args()
    return args


class LidarCamPub:
    def __init__(self, publish_rate_divisor):
        rospy.init_node('lidar_cam_pub', anonymous=False)
        self.pointcloud_sub = rospy.Subscriber('velodyne_points', PointCloud2, self.callback, queue_size=10)
        self.processed_pointcloud_pub = rospy.Publisher('processed_points', PointCloud2, queue_size=10)
        if PUB_IMAGE:
            self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback, queue_size=10)
            self.synced_image_pub = rospy.Publisher('synced_image', Image, queue_size=2)

        self.marker_pub = rospy.Publisher('angle_boundaries', Marker, queue_size=10)
        self.marker = self.create_angle_boundaries()

        self.callback_count = 0
        self.img_msg = None
        self.publish_rate_divisor = publish_rate_divisor  # 从命令行参数获取

    def image_callback(self, img_msg):
        self.img_msg = img_msg

    def callback(self, pcl_msg):
        self.callback_count += 1

        if self.callback_count >= self.publish_rate_divisor:  # 每5条消息处理一次  10Hz -> 2Hz
            self.callback_count = 0

            # 进行点云处理
            processed_points = self.convert_pointcloud2(pcl_msg)

            # 发布处理后的点云数据
            self.publish_processed_pointcloud(processed_points)

            if PUB_IMAGE:
                if self.img_msg is None:
                    rospy.logerr("No image message received.")
                    return

                # 发布图像
                self.synced_image_pub.publish(self.img_msg)

            # 发布 Marker
            self.marker.header.stamp = rospy.Time.now()
            self.marker_pub.publish(self.marker)

            rospy.loginfo("All messages published.")

    def convert_pointcloud2(self, pcl_msg):
        """
        将PointCloud2点云数据转换为模型输入格式, 并进行ROI过滤
        @param pcl_msg  PointCloud2  ROS消息
        @return:        np.ndarray  [N, 4]  处理后的点云数据
        """
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

        # 限制x坐标距离
        x_distance_limit = 30

        # 过滤点：只保留前方并在角度限制内且x坐标在距离限制内的点
        mask = (x > 1) & (x < x_distance_limit) & (y >= -x_offset * right_tan) & (y <= x_offset * left_tan)

        # 应用掩膜过滤点并返回
        points = np.column_stack((x[mask], y[mask], z[mask], i[mask]))
        return points

    def create_pointcloud2_msg(self, points):
        """
        从处理后的点云数据创建PointCloud2消息
        @param points  np.ndarray  [N, 4]  处理后的点云数据
        @return:       PointCloud2  ROS消息
        """
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
        @param processed_points  np.ndarray  [N, 4]  处理后的点云数据
        """
        point_cloud_msg = self.create_pointcloud2_msg(processed_points)

        # 发布消息
        self.processed_pointcloud_pub.publish(point_cloud_msg)

    def create_angle_boundaries(self):
        """
        设置过滤范围边界线消息
        """
        # 计算边界线的端点
        line_length = 7  # 设置线长
        x_origin = 1.0  # 设置原点
        left_angle = math.radians(45)
        right_angle = math.radians(45)  # 设置角度, 约为相机可视范围
        left_boundary = (x_origin + line_length, line_length * math.tan(left_angle))  # x + d
        right_boundary = (x_origin + line_length, -line_length * math.tan(right_angle))  # x + d

        # 创建Marker
        marker = Marker()
        marker.header.frame_id = "velodyne"
        marker.ns = "angle_boundaries"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # 设置位姿
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # 设置线条宽度
        marker.scale.x = 0.1

        # 设置线条颜色rgba
        marker.color.r = 1.0  # 红色
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # 不透明

        # 设置 Marker points
        marker.points.append(Point(x_origin, 0, 0))  # 起点
        marker.points.append(Point(left_boundary[0], left_boundary[1], 0))  # 左边界
        marker.points.append(Point(x_origin, 0, 0))  # 起点
        marker.points.append(Point(right_boundary[0], right_boundary[1], 0))  # 右边界
        return marker


if __name__ == "__main__":
    try:
        args = parse_arguments()
        print("publish_rate_divisor:", args.r)
        lidar_cam_pub = LidarCamPub(args.r)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
