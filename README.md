# 基于激光雷达与图像信息融合的道路目标检测

![可视化](https://github.com/livvta/LidarImageFusionDetection/blob/master/README_PIC/图片2.png)

## 运行方式

0. 项目运行的一切建立在ROS上，所以 ```roscore，启动！```
   
   ```
   roscore
   ```

1. 请选择合适的方式发布数据（```PubTools```or```播包```or```硬件驱动```）
   
   ```
   # 启动 PubTools
   cd ~/detection_ws/src/publish_utils/scripts
   python2 pubtools.py
   ```
   
   ```
   # 播包
   # 0.5倍播包请加 -r 0.5
   rosbag play 2025-02-23-16-35-54.bag --loop -r 0.5 /usb_cam/image_raw:=/camera/image_raw
   rosbag play 2025-02-23-16-35-54.bag --loop -r 0.5 /velodyne_points:=/points_raw
   ```
   
   ```
   # 启动 Velodyne VLP-16 激光雷达
   roslaunch velodyne_pointcloud VLP16_points.launch
   ```
   
   ```
   # 启动 usb_cam
   roslaunch usb_cam usb_cam-test.launch
   ```
   
   启动驱动时可能会遇到问题 请检查并排除以下常见原因  
   1. 驱动配置文件（分辨率、刷新率、消息话题）  
   2. USB拓展坞  

2. 启动数据预处理及频率控制模块
   
   ```
   cd ~/detection_ws/src/publish_utils/scripts
   python2 lidar_cam_pub.py
   ```

3. 启动pointpillars节点
   
   ```
   conda activate pytorch12
   rosrun pointpillars_ros pointpillars_node.py
   ```

4. 启动yolov5节点
   
   ```
   conda activate yolov5
   rosrun yolov5_ros yolov5_node.py
   ```

5. 启动融合模块
   
   ```
   conda activate pytorch12
   cd ~/detection_ws/src/fusion/scripts
   python fusion_decision.py -iou_thresh=0.4 -pp_weight=1.0 -yolo_weight=0.9
   ```

### 常见问题

1. Autoware联合标定
   
   ```
   # 启动 Autoware 联合标定
   cd catkin_ws
   source devel/setup.bash
   rosrun calibration_camera_lidar calibration_toolkit
   ```
   
   ###### 注意！
   
   工程中心实验室 ```带有三个圆孔的棋盘格标定板 @吴昊```与Autoware标定工具不适配，不可使用。
   
   标定时保证光线充足，否则会造成棋盘格角点检测失败及偏移。
   
   

2. 如果移动了mmdetection3d源码的位置需要重新编译一下
   
   ```
   cd mmdetection3d
   mim install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. 休眠后torch报错
   
   ```
   sudo rmmod nvidia_uvm
   sudo modprobe nvidia_uvm
   ```

4. 绘制图像
   
   ```
   # 绘制PointPillars_KITTI16 loss曲线
   python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/pointpillars_hv_secfpn_8xb4-160e_kitti16-3d-3class/20241127_205454/vis_data/20241127_205454.json --keys loss_cls loss_bbox  --legend loss_cls_PointPillars_KITTI16  loss_bbox_PointPillars_KITTI16
   
   # 绘制PointPillars_KITTI64 loss曲线
   python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/pointpillars_hv_secfpn_8xb4-160e_kitti-3d-3class/20241012_201716/vis_data/20241012_201716.json --keys loss_cls loss_bbox --legend loss_cls_PointPillars_KITTI64 loss_bbox_PointPillars_KITTI64
   ```

#### 更详细的请见代码注释，祝好运！
![实车](https://github.com/livvta/LidarImageFusionDetection/blob/master/README_PIC/图片1.png)
<br><br>
> Updated on May 13, 2025  
> 李浩洋 LI HAOYANG  
> 山东交通学院 轨道交通学院 Shandong Jiaotong University  
