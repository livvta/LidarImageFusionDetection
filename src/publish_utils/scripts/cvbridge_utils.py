#!/usr/bin/env python2
# coding:utf-8

import sys
import cv2
import os
import numpy as np
import rospy 
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2


def imgmsg_to_cv2(img_msg):
    '''
    将ROS Image消息转换为OpenCV格式
    @param img_msg      ROS Image消息
    @return:            np.ndarray  OpenCV格式图像
    ''' 
    if img_msg.encoding != "bgr8":
        raise ValueError("Unsupported image encoding: {}".format(img_msg.encoding))
    dtype = np.dtype("uint8").newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(
        shape=(img_msg.height, img_msg.width, 3), 
        dtype=dtype, 
        buffer=img_msg.data
    )
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()

    return image_opencv


def cv2_to_imgmsg(cv_image):
    '''
    将OpenCV格式图像转换为ROS Image消息
    @param cv_image     np.ndarray  OpenCV格式图像
    @return:            ROS Image消息
    '''
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg
