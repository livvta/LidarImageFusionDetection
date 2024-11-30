#!/usr/bin/env python2
# coding:utf-8

'''
camear_pub.py
==================
Version: 1.0
Last Modified: 2024-12-01 00:03

功能：
- 读取摄像头数据并发布到/camera/image_raw话题
'''

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def publish_camera():
    rospy.init_node('camera_publisher', anonymous=True) # 待定
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    bridge = CvBridge()
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/dev/video2')

    if not cap.isOpened():
        rospy.logerr("Cannot open camera")
        return

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Failed to capture image")
            break

        image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_pub.publish(image_msg)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        publish_camera()
    except rospy.ROSInterruptException:
        pass
