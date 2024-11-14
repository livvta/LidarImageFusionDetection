#!/usr/bin/env python2
# coding:utf-8

'''
single_image_pub
================

Version: 1.0.0
Last Modified: 2024-10-03

此脚本用于发布指定图像到 ROS 话题 /camera/image_raw

警告: 此程序基于python2.7构建,python3可能存在未知兼容性问题
'''

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def single_image_pub():
    # 初始化ROS节点
    rospy.init_node('single_image_pub', anonymous=False)
    
    # 创建/camera/image_raw话题，消息类型为Image
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

    # 使用cv_bridge
    bridge = CvBridge()

    # 读取一张图片
    frame = cv2.imread('/home/harris/detection_ws/src/yolov5_ros/scripts/dataset/kitti/images/train/000031.png')

    if frame is None:
        rospy.logerr("无法读取图片")
        return
    
    # 发布图片一次
    try:
        # 将OpenCV的图像转换为ROS的Image消息
        image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        
        # 发布图像消息
        image_pub.publish(image_msg)
        
        rospy.loginfo("发布相机画面")
        
    except CvBridgeError as e:
        rospy.logerr("cv_bridge转换错误: %s", e)
    
    # 休眠以保持节点活动
    rospy.sleep(1)

if __name__ == '__main__':
    try:
        single_image_pub()
    except rospy.ROSInterruptException:
        pass
