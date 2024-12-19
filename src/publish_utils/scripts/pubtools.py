#!/usr/bin/env python2
# coding:utf-8

"""
pubtools
==============================
Version: 1.1
Created Time: 2024-11-25

Update:
V1.1: 241126优化代码以减少重复并提高可读性
"""

import cv2
import os
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
from cvbridge_utils import cv2_to_imgmsg


def pointcloud_pub(point_cloud, pcl_pub):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'velodyne'
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
    ]
    cloud_msg = pcl2.create_cloud(header, fields, point_cloud)
    pcl_pub.publish(cloud_msg)


def image_pub(img, image_pub):
    img_msg = cv2_to_imgmsg(img)
    image_pub.publish(img_msg)


def publish_data(image_path, pc_path, auto_mode=True):
    frame = 0 if auto_mode else int(raw_input("Enter starting frame: "))
    rate = rospy.Rate(int(raw_input("Enter Rate: "))) if auto_mode else None

    pcl_pub = rospy.Publisher('velodyne_points', PointCloud2, queue_size=10)
    cam_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

    pcl_files = sorted([f for f in os.listdir(pc_path) if f.endswith('.bin')])
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    while not rospy.is_shutdown():
        point_cloud = np.fromfile(os.path.join(pc_path, pcl_files[frame]), dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(os.path.join(image_path, image_files[frame]))

        if img is None:
            rospy.logerr("无法读取图片: %s", os.path.join(image_path, image_files[frame]))
            break

        pointcloud_pub(point_cloud, pcl_pub)
        image_pub(img, cam_pub)

        rospy.loginfo("Image & PointCloud published: %d", frame)

        if not auto_mode:
            cv2.destroyAllWindows()
            cv2.imshow("image: {}".format(frame), img)
            key = cv2.waitKey(0)
            if key == ord('m'):  # 下一帧
                frame = (frame + 1) % len(image_files)
            elif key == ord('n'):  # 上一帧
                frame = (frame - 1) % len(image_files)
            elif key == ord('q'):  # 退出
                break
        else:
            rate.sleep()
            frame = (frame + 1) % len(image_files)

    cv2.destroyAllWindows()


def manual_image_pub(image_path):
    frame = int(raw_input("Enter starting frame: "))
    cam_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    while not rospy.is_shutdown():
        img = cv2.imread(os.path.join(image_path, image_files[frame]))

        if img is None:
            rospy.logerr("无法读取图片: %s", os.path.join(image_path, image_files[frame]))
            break

        image_pub(img, cam_pub)
        rospy.loginfo("Image published: %d", frame)

        cv2.destroyAllWindows()
        cv2.imshow("image: {}".format(frame), img)

        key = cv2.waitKey(0)
        if key == ord('m'):
            frame = (frame + 1) % len(image_files)
        elif key == ord('n'):
            frame = (frame - 1) % len(image_files)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


def manual_pointcloud_pub(pc_path):
    """
    手动依次发布点云到 ROS 话题 /velodyne_points
    """
    frame = int(raw_input("Enter starting frame: "))
    pcl_pub = rospy.Publisher('velodyne_points', PointCloud2, queue_size=10)

    image_files = sorted([f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    pcl_files = sorted([f for f in os.listdir(pc_path) if f.endswith(('.bin'))])

    while not rospy.is_shutdown():
        point_cloud = np.fromfile(os.path.join(pc_path, pcl_files[frame]), dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(os.path.join(image_path, image_files[frame]))
        # resize image
        img_resized = cv2.resize(img, (621, 187))

        pointcloud_pub(point_cloud, pcl_pub)
        rospy.loginfo("PointCloud published: %d", frame)

        cv2.destroyAllWindows()
        cv2.imshow("Pointcloud2 Publisher No. {}".format(frame), img_resized)

        key = cv2.waitKey(0)
        if key == ord('m'):  # 按下 'm' 键切换到下一张
            frame = (frame + 1) % len(image_files)
        elif key == ord('n'):  # 按下 'n' 键切换到上一张
            frame = (frame - 1) % len(image_files)
        elif key == ord('q'):  # 按下 'q' 键退出
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('kitti_publisher', anonymous=True)

    IMAGE_KITTI_RAW = '/media/harris/PM981/kitti_16/testing/image_2'
    PC_KITTI_RAW_16 = '/media/harris/PM981/kitti_16/testing/velodyne'
    IMAGE_09260005 = '/home/harris/dataset/RawData/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/'
    PC_09260005_64 = '/home/harris/dataset/RawData/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/'
    PC_09260005_16 = '/home/harris/dataset/RawData/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/d_data/'

    print("                                             ")
    print("  ____        _     _____            _       ")
    print(" |  _ \ _   _| |__ |_   _|___   ___ | | ___  ")
    print(" | |_) | | | | '_ \  | | / _ \ / _ \| |/ __| ")
    print(" |  __/| |_| | |_) | | || (_) | (_) | |\__ \ ")
    print(" |_|    \__,_|_.__/  |_| \___/ \___/|_||___/ ")
    print("                                             ")

    print("请选择要发布的数据集:")
    print("1. 09260005_64线")
    print("2. 09260005_16线")
    print("3. KITTI_RAW_16线")

    dataset_select = raw_input("Enter: ")

    if dataset_select == '1':
        image_path = IMAGE_09260005
        pc_path = PC_09260005_64
    elif dataset_select == '2':
        image_path = IMAGE_09260005
        pc_path = PC_09260005_16
    elif dataset_select == '3':
        image_path = IMAGE_KITTI_RAW
        pc_path = PC_KITTI_RAW_16

    print("======================================")
    print("请选择发布方式:")
    print("1. 自动顺序发布Image, PointCloud")
    print("2. 手动顺序发布Image, PointCloud")
    print("3. 手动顺序发布Image")
    print("4. 手动顺序发布PointCloud")

    function_select = raw_input("Enter: ")

    if function_select == '1':
        publish_data(image_path, pc_path, auto_mode=True)
    elif function_select == '2':
        publish_data(image_path, pc_path, auto_mode=False)
    elif function_select == '3':
        manual_image_pub(image_path)
    elif function_select == '4':
        manual_pointcloud_pub(pc_path)
