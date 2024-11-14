#!/usr/bin/env python2
#coding:utf-8

'''
camera_pub
================

Version: 1.0.0
Last Modified: 2024-10-03

警告: 此程序基于python2.7构建,python3可能存在未知兼容性问题
'''

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def camera_publisher():
    # 初始化ROS节点
    rospy.init_node('camera_pub', anonymous=False)
    
    # 创建/camera/image_raw话题，消息类型为Image
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

    # 设置循环频率
    rate = rospy.Rate(10)  # 10Hz
    
    # 使用cv_bridge将OpenCV的图像转换为ROS的Image消息类型
    bridge = CvBridge()
    
    # 打开相机
    # cap = cv2.VideoCapture(0)

    # or 读取一段视频
    cap = cv2.VideoCapture('/home/harris/Downloads/pov1.mp4')


    # 检查相机是否打开成功
    if not cap.isOpened():
        rospy.logerr("无法打开相机")
        return
    
    while not rospy.is_shutdown():
        # 读取相机画面
        ret, frame = cap.read()
        
        if not ret:
            rospy.logerr("无法读取相机画面")
            break
        
        try:
            # 将OpenCV的图像转换为ROS的Image消息
            image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            
            # 发布图像消息
            image_pub.publish(image_msg)
            
            # 打印日志信息
            # rospy.loginfo("发布相机画面")
        except CvBridgeError as e:
            rospy.logerr("cv_bridge转换错误: %s", e)
        
        # 让循环按照指定频率执行
        rate.sleep()
    
    # 关闭相机
    cap.release()

if __name__ == '__main__':
    try:
        camera_publisher()
    except rospy.ROSInterruptException:
        pass