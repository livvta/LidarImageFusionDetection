#!/usr/bin/env python2
# coding:utf-8

'''
kitti_raw_date_pub
================

Version: 1.3
Last Modified: 2024-11-09 22:56

警告: 此程序基于python2.7构建,python3可能存在未知兼容性问题
'''

import cv2
import os
import numpy as np
import rospy 
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge

DATA_PATH = '/home/harris/dataset/RawData/2011_09_26/2011_09_26_drive_0005_sync/'

if  __name__ == "__main__":
    frame = 0
    rospy.init_node('kitti_node',anonymous=False)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('velodyne_points', PointCloud2, queue_size=10)
    bridge = CvBridge()

    str = input("Enter Rate: ")
    rate = rospy.Rate(str)

    while not rospy.is_shutdown():

        img = cv2.imread(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%frame))
        img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
        # img_msg.header.frame_id = 'camera_link'
        # img_msg.header.stamp = rospy.Time.now()
        cam_pub.publish(img_msg)

        point_cloud = np.fromfile(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%frame),dtype=np.float32).reshape(-1,4)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'velodyne'

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]

        # 发布带强度的点云
        cloud_msg = pcl2.create_cloud(header, fields, point_cloud)

        pcl_pub.publish(cloud_msg)

        rospy.loginfo("Kitti published")


        rate.sleep()
        frame += 1
        frame %= 154

        
