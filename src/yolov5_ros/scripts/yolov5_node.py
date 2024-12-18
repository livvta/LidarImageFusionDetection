#!/usr/bin/env python
# encoding=utf-8

'''
yolov5_node
================

Version: 2.3
Last Modified: 2024-12-28

更新日志: 
11.05: 将YOLODetection消息类型更新为BoundingBox2DArray
11.08: 提高代码复用, 优化可视化效果
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

# 是否启用可视化功能
rviz_visualization = True

# 设置置信度阈值（仅对可视化有效)
yolo_confidence_threshold = 0.5

# 1, 2标签互换
label_map = {1: 2, 2: 1}
# 0: pedestrian
# 1: cyclist
# 2: car

class YoloV5Node:
    def __init__(self):
        # 初始化模型
        self.yolov5_model = torch.hub.load(dir, model, path=path, source=source)

        # 初始化 ROS 节点, 节点名称为yolov5_node
        rospy.init_node('yolov5_node', anonymous=False)
        
        # 订阅图像话题, 消息类型为Image
        self.image_sub = rospy.Subscriber("synced_image", Image, self.image_callback)
        
        # 创建yolo_results话题, 消息类型为BoundingBox2DArray, 用于发布检测结果
        self.yolo_detection_pub = rospy.Publisher('yolo_results', BoundingBox2DArray, queue_size=10)

        # 创建yolo_visualization话题，消息类型Image，用于在rviz中可视化检测结果
        self.yolo_visualization_pub = rospy.Publisher('yolo_visualization', Image, queue_size=10)

    def image_callback(self, msg):
        try:
            # 将Image消息转换为模型输入格式
            cv_image = self.imgmsg_to_cv2(msg)

            # 调用YoloV5模型检测
            results = self.yolov5_model(cv_image)

            # 发布检测结果
            self.publish_results(results)

            # 可视化YOLO检测结果
            if rviz_visualization:
                image_copy = self.draw_detections(cv_image, results)
                img_msg = self.cv2_to_imgmsg(image_copy)
                self.yolo_visualization_pub.publish(img_msg)

        except Exception as e:
            rospy.logerr("YoloV5Node error: %s", e)
            return


    def imgmsg_to_cv2(self, img_msg):
        '''
        将ROS Image消息转换为OpenCV格式
        @param img_msg:      ROS Image消息
        @return:             np.ndarray  OpenCV格式图像
        ''' 
        # 检查图像编码格式
        if img_msg.encoding != "bgr8":
            raise ValueError("Unsupported image encoding: {}".format(img_msg.encoding))
        
        # 创建图像数据类型
        dtype = np.dtype("uint8").newbyteorder('>' if img_msg.is_bigendian else '<')

        # 使用缓冲区创建 OpenCV 图像
        image_opencv = np.ndarray(
            shape=(img_msg.height, img_msg.width, 3), 
            dtype=dtype, 
            buffer=img_msg.data
        )
        
        return image_opencv


    def cv2_to_imgmsg(self, cv_image):
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
        img_msg.data = cv_image.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height
        return img_msg


    def draw_detections(self, image, results):
        '''
        绘制YOLO检测结果
        @param image:       np.ndarray  OpenCV格式图像
        @param results:     Det2DDataSample  YOLO检测结果
        '''
        # 复制原图像
        image_copy = image.copy()

        colors = [
            (0, 150, 180),    # 黄色, pedestrian
            (255, 0, 0),      # 蓝色, car
            (158, 0, 150),    # 紫色, cyclist
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX # 字体
        thickness = 1       # 标签字体粗细
        font_size = 0.5     # 标签字号

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf < yolo_confidence_threshold: 
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

        # cv2.imshow('YOLOv5 Detection', image_copy)
        # cv2.waitKey(1)
        return image_copy

    def publish_results(self, results):
        '''
        以BoundingBox2DArray消息类型发布YOLO检测结果
        @param results:     Det2DDataSample  YOLO检测结果
        -------------------------------------------------
        yolov5_ros.msg/BoundingBox2D.msg
        float32 x_min // 左上角x坐标
        float32 y_min // 左上角y坐标
        float32 x_max // 右下角x坐标
        float32 y_max // 右下角y坐标
        float32 value // 置信度
        uint32 label  // 类别
        -------------------------------------------------
        yolov5_ros.msg/BoundingBox2DArray.msg
        Header header // ROS消息头
        BoundingBox2D[] boxes
        '''
        bbox2d_array = BoundingBox2DArray()
        bbox2d_array.header.stamp = rospy.Time.now()
        bbox2d_array.header.frame_id = "camera_link"

        for *xyxy, conf, cls in results.xyxy[0]:
            # 解析边界框参数           
            x_min, y_min, x_max, y_max = xyxy
            label = label_map.get(int(cls), int(cls)) # 使用字典映射

            bbox2d = BoundingBox2D()
            bbox2d.x_min = float(x_min)
            bbox2d.y_min = float(y_min)
            bbox2d.x_max = float(x_max)
            bbox2d.y_max = float(y_max)
            bbox2d.value = float(conf)
            bbox2d.label = label
            bbox2d.model = 0
            
            bbox2d_array.boxes.append(bbox2d)

        self.yolo_detection_pub.publish(bbox2d_array)




if __name__ == '__main__':
    try:
        node = YoloV5Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()

