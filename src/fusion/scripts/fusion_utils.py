import numpy as np
import sensor_msgs.point_cloud2 as pcl2

# 标定文件路径
calib_path = '/home/harris/detection_ws/src/fusion/scripts/calibration.txt'

def read_calib():
    '''
    读取标定文件
    '''
    with open(calib_path, 'r') as f:
        raw = f.readlines()
    P0 = np.array(list(map(float, raw[0].split()[1:]))).reshape((3, 4))
    P1 = np.array(list(map(float, raw[1].split()[1:]))).reshape((3, 4))
    P2 = np.array(list(map(float, raw[2].split()[1:]))).reshape((3, 4))
    P3 = np.array(list(map(float, raw[3].split()[1:]))).reshape((3, 4))
    R0 = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 3))
    R0 = np.hstack((R0, np.array([[0], [0], [0]])))
    R0 = np.vstack((R0, np.array([0, 0, 0, 1])))
    lidar2camera_m = np.array(list(map(float, raw[5].split()[1:]))).reshape((3, 4))
    lidar2camera_m = np.vstack((lidar2camera_m, np.array([0, 0, 0, 1])))
    imu2lidar_m = np.array(list(map(float, raw[6].split()[1:]))).reshape((3, 4))
    imu2lidar_m = np.vstack((imu2lidar_m, np.array([0, 0, 0, 1])))
    return P0, P1, P2, P3, R0, lidar2camera_m, imu2lidar_m


def imgmsg_to_cv2(img_msg):
    '''
    将Imagea消息转换为opencv_image
    '''
    if img_msg.encoding != "bgr8":
        raise ValueError("Unsupported image encoding: {}".format(img_msg.encoding))
    
    # 创建图像数据类型
    dtype = np.dtype("uint8").newbyteorder('>' if img_msg.is_bigendian else '<')

    # 创建 OpenCV 图像
    image_opencv=np.ndarray(
        shape=(img_msg.height, img_msg.width, 3), 
        dtype=dtype, 
        buffer=img_msg.data
    )
    
    return image_opencv


def process_points(pcl_msg):
    '''
    读取并处理点云消息
    '''
    pcl_data = pcl2.read_points(pcl_msg, field_names=("x", "y", "z"), skip_nans=True)
    pcl_array = np.array(list(pcl_data))

    return pcl_array

# # 将3d框投影到2d平面上
# def project_bounding_boxes(self, bounding_boxes, K, lidar_to_camera, imu_to_lidar):
#     projected_boxes = []
    
#     for box in bounding_boxes:
#         # 从字典中获取中心位置和尺寸
#         position = box['position']
#         dimensions = box['dimensions']
        
#         # 计算 3D 边界框的八个顶点
#         corners = self.get_box_corners(position, dimensions)
        
#         # 将点从激光雷达坐标系转换到相机坐标系
#         corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1))))
#         transformed_corners = (lidar_to_camera @ (imu_to_lidar @ corners_homogeneous.T)).T
        
#         # 投影到 2D 平面
#         projected_corners = (K @ transformed_corners.T).T
#         projected_corners /= projected_corners[:, 2].reshape(-1, 1)  # 归一化
        
#         # print(projected_corners)

#         # 找到 2D 边界框的最小和最大坐标
#         u_min, u_max = np.min(projected_corners[:, 0]), np.max(projected_corners[:, 0])
#         v_min, v_max = np.min(projected_corners[:, 1]), np.max(projected_corners[:, 1])
        
#         projected_boxes.append((u_min, v_min, u_max, v_max))
    
#     return projected_boxes

