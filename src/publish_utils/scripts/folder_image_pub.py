#!/usr/bin/env python2
# coding:utf-8

'''
folder_image_pub
================

Version: 1.2.0
Last Modified: 2024-10-04 16:42

此脚本用于从指定文件夹，以文件名称为顺序，依次发布图像到 ROS 话题 /camera/image_raw
适用于测试kitti数据集训练结果

在opencv窗口，按下 'n' 键切换到上一张，按下 'm' 键切换到下一张，按下'q'退出程序

警告: 此程序基于python2.7构建,python3可能存在未知兼容性问题
'''

import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def folder_image_pub(image_folder):
    # 初始化ROS节点
    rospy.init_node('folder_image_pub', anonymous=False)
    
    # 创建/camera/image_raw话题，消息类型为Image
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

    # 使用cv_bridge
    bridge = CvBridge()

    # 获取文件夹内的所有图片文件名并排序
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        rospy.logerr("文件夹内没有图片")
        return

    index = 0  # 当前图片索引

    while not rospy.is_shutdown():
        # 读取当前图片
        image_path = os.path.join(image_folder, image_files[index])
        frame = cv2.imread(image_path)

        if frame is None:
            rospy.logerr("无法读取图片: %s", image_path)
            break
        
        try:
            # 将OpenCV的图像转换为ROS的Image消息
            image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            
            # 发布图像消息
            image_pub.publish(image_msg)

            cv2.destroyAllWindows()  # 关闭所有窗口
            rospy.loginfo("发布图片: %s", image_files[index])

        except CvBridgeError as e:
            rospy.logerr("cv_bridge转换错误: %s", e)

        cv2.imshow("image: {}".format(image_files[index]),frame)

        # 等待用户输入切换图片
        key = cv2.waitKey(0)  # 等待按键
        if key == ord('m'):  # 按下 'm' 键切换到下一张
            index = (index + 1) % len(image_files)  # 循环切换到下一张
        elif key == ord('n'):  # 按下 'n' 键切换到上一张
            index = (index - 1) % len(image_files)  # 循环切换到上一张
        elif key == ord('q'):  # 按下 'q' 键退出
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_folder = '/home/harris/dataset/kitti/images/train'  # 替换为你的图片文件夹路径
    try:
        folder_image_pub(image_folder)
    except rospy.ROSInterruptException:
        pass
