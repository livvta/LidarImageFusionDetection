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


