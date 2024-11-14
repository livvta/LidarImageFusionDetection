#!/usr/bin/env python
# encoding=utf-8

'''
yolov5_node
================

Version: 2.2
Last Modified: 2024-11-08 01:16

更新日志：
11.05：将YOLODetection消息类型更新为BoundingBox2DArray
11.08：提高代码复用，优化可视化效果
'''

import cv2
import rospy
import numpy as np
import torch
from sensor_msgs.msg import Image
from yolov5_ros.msg import BoundingBox2D, BoundingBox2DArray

# 模型配置信息
dir = '/home/harris/model/yolov5'
model = 'custom'
path = '/home/harris/model/yolov5/runs/train/exp7/weights/best.pt'
source = 'local'

# 初始化模型
model = torch.hub.load(dir, model, path=path, source=source)

# 是否启用可视化功能
cv2_visualization = True

# 设置置信度阈值（仅对可视化有效)
confidence_threshold = 0.5


class YoloV5Node:
    def __init__(self):
        # 初始化 ROS 节点 , 节点名称为 yolov5_node
        rospy.init_node('yolov5_node', anonymous=False)
        
        # 订阅图像话题 , 话题名称为 /camera/image_raw
        self.image_sub = rospy.Subscriber("kitti_cam", Image, self.image_callback)
        
        # 创建yolo_detections话题，消息类型为BoundingBox2DArray，用于发布检测结果
        self.yolo_detection_pub = rospy.Publisher('yolo_detections', BoundingBox2DArray, queue_size=10)


    def image_callback(self, msg):
        try:
            # 将Image消息转换为cv2
            cv_image = self.imgmsg_to_cv2(msg)

            # 调用模型进行检测
            results = model(cv_image)

            # 发布检测结果
            self.publish_results(results)

            # 可视化YOLO检测结果
            if cv2_visualization:
                self.draw_detections(cv_image, results)
          
        except Exception as e:
            rospy.logerr("YoloV5Node error: %s", e)
            return


    def imgmsg_to_cv2(self, img_msg):
        '''因 CV_Bridge 与 Python 3 兼容性差，须手动实现转换。 ''' 
        # 检查图像编码格式
        if img_msg.encoding != "bgr8":
            raise ValueError("Unsupported image encoding: {}".format(img_msg.encoding))
        
        # 创建图像数据类型
        dtype = np.dtype("uint8").newbyteorder('>' if img_msg.is_bigendian else '<')

        # 使用缓冲区创建 OpenCV 图像;
        image_opencv=np.ndarray(
            shape=(img_msg.height, img_msg.width, 3), 
            dtype=dtype, 
            buffer=img_msg.data
        )
        
        return image_opencv


    def draw_detections(self, image, results):
        # 设置类别颜色
        colors = [
            (255, 0, 0),    # 红色
            (0, 190, 0),    # 深绿色
            (0, 0, 255),    # 蓝色
        ]
        thickness = 1       # 标签粗细
        font_size = 0.5     # 标签字号
        font = cv2.FONT_HERSHEY_SIMPLEX # 字体

        image_copy = image.copy()

        for *xyxy, conf, cls in results.xyxy[0]:

            if conf < confidence_threshold: 
                continue

            x_min, y_min, x_max, y_max = map(int, xyxy)
            cls = int(cls)
            class_name = results.names[cls]
            color = tuple(colors[cls]) 
            label = f"{class_name} {conf:.2f}"

            # 绘制边界框            
            cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, 2)

            # 绘制标签背景
            label_size = cv2.getTextSize(label, font, font_size, thickness)[0]
            label_w, label_h = label_size
            label_x = x_min - 1
            label_y = y_min - label_h - 1
            cv2.rectangle(image_copy, (label_x, label_y), (label_x + label_w, label_y + label_h), color, -1)

            # 绘制文字标签
            cv2.putText(image_copy, label, (x_min, y_min - 2), font, font_size, (255, 255, 255), thickness)

        cv2.imshow('YOLOv5 Detection', image_copy)
        cv2.waitKey(1)

        return


    def publish_results(self, results):
        '''将检测结果处理为BoundingBox2DArray类型'''
        bbox2d_array = BoundingBox2DArray()
        bbox2d_array.header.stamp = rospy.Time.now()
        bbox2d_array.header.frame_id = "camera_link" 

        # bboxes = []
        # value = []
        # class_names = []

        for *xyxy, conf, cls in results.xyxy[0]:
            x_min, y_min, x_max, y_max = xyxy

            bbox2d = BoundingBox2D()

            bbox2d.x_min = float(x_min)
            bbox2d.y_min = float(y_min)
            bbox2d.x_max = float(x_max)
            bbox2d.y_max = float(y_max)
            bbox2d.value = float(conf)
            bbox2d.label = int(cls)
            
            bbox2d_array.boxes.append(bbox2d)

            self.yolo_detection_pub.publish(bbox2d_array)

        return


if __name__ == '__main__':
    try:
        node = YoloV5Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass

