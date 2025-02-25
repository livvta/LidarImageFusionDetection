import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pcl2

# 标定文件路径
calib_path = '/home/harris/detection_ws/src/fusion/scripts/calibration_apollo.txt'


def read_calib():
    """
    读取标定文件
    """
    with open(calib_path, 'r') as f:
        raw = f.readlines()
    P0 = np.array(list(map(float, raw[0].split()[1:]))).reshape((3, 4))
    P1 = np.array(list(map(float, raw[1].split()[1:]))).reshape((3, 4))
    P2 = np.array(list(map(float, raw[2].split()[1:]))).reshape((3, 4))  # 此为KITTI相机内参
    P3 = np.array(list(map(float, raw[3].split()[1:]))).reshape((3, 4))
    R0 = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 3))
    R0 = np.hstack((R0, np.array([[0], [0], [0]])))
    R0 = np.vstack((R0, np.array([0, 0, 0, 1])))
    lidar2camera_m = np.array(list(map(float, raw[5].split()[1:]))).reshape((3, 4))
    lidar2camera_m = np.vstack((lidar2camera_m, np.array([0, 0, 0, 1])))
    imu2lidar_m = np.array(list(map(float, raw[6].split()[1:]))).reshape((3, 4))
    imu2lidar_m = np.vstack((imu2lidar_m, np.array([0, 0, 0, 1])))

    # Attetion！！！仅在Apollo实验时启用以下代码
    P2 = np.array(list(map(float, raw[7].split()[1:]))).reshape((3, 4))  # 此为Apollo摄像头内参
    apollo_extrinsic = np.array(list(map(float, raw[8].split()[1:]))).reshape((3, 4))  # 此为Apollo摄像头外参
    apollo_extrinsic = np.vstack((apollo_extrinsic, np.array([0, 0, 0, 1])))  # 此为Apollo摄像头外参
    return P0, P1, P2, P3, R0, lidar2camera_m, imu2lidar_m, apollo_extrinsic


def calibration_param():
    """
    计算标定内外参数
    """
    P0, P1, P2, P3, R0, lidar2camera_matrix, imu2lidar_matrix, apollo_extrinsic= read_calib()
    intrinsic = P2[:, :3]  # KITTI Cam 2(color)
    """
                    [fx   0  cx]
        intrinsic = [ 0  fy  cy]  fx, fy:相机焦距
                    [ 0   0   1]  cx, cy:相机主点
    """
    extrinsic = np.matmul(R0, lidar2camera_matrix)
    """
        extrinsic = [R | T]       R:旋转矩阵[3, 3]
                                  T:平移向量[3, 1]
    """

    # Attetion！！！仅在Apollo实验时启用以下代码
    intrinsic = P2  # Only for Apollo
    extrinsic = apollo_extrinsic  # Only for Apollo
    print("R0\n", R0)
    print("intrinsic\n", intrinsic)
    print("extrinsic\n", extrinsic)
    return intrinsic, extrinsic

def imgmsg_to_cv2(img_msg):
    """
    将ROS Image消息转换为OpenCV格式
    @param img_msg:      ROS Image消息
    @return:             np.ndarray  OpenCV格式图像
    """

    if img_msg.encoding not in ["bgr8", "rgb8"]:  # 检查图像编码格式
        raise ValueError("Unsupported image encoding: {}".format(img_msg.encoding))
    dtype = np.dtype("uint8").newbyteorder('>' if img_msg.is_bigendian else '<')  # 创建图像数据类型
    image_opencv = np.ndarray(
        shape=(img_msg.height, img_msg.width, 3),
        dtype=dtype,
        buffer=img_msg.data
    )  # 使用缓冲区创建 OpenCV 图像
    if img_msg.encoding == "rgb8":
        image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    """
    将OpenCV格式图像转换为ROS Image消息
    @param cv_image     np.ndarray  OpenCV格式图像
    @return:            ROS Image消息
    """
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tobytes()
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg

def process_points(pcl_msg):
    """
    读取并处理点云消息
    """
    pcl_data = pcl2.read_points(pcl_msg, field_names=("x", "y", "z"), skip_nans=True)
    pcl_array = np.array(list(pcl_data))
    return pcl_array

def bbox3d_center_to_corners(bboxes):
    """
    通过边界框的位置(底面中点)、尺寸、朝向角参数得到边界框的8个角点坐标
    box_3d: (x, y, z, x_size, y_size, z_size, yaw)
    @param bboxes:      np.ndarray  [N, 7]
    @return:            np.ndarray  [N, 8, 3]

           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |
    y      | /              |  |     | |
    <------|o               | 7 -----| 4
                            |/   o   |/
                            3 ------ 0
    x: front, y: left, z: top
    """
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
    # 1. 计算边界框顶点坐标, 按照顺时针方向从最小点排列 （3, 0, 4, 7, 2, 1, 5, 6）
    bboxes_corners = np.array([[-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0],
                                [-0.5, 0.5, 1.0], [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0]],
                                dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]  # [1, 8, 3] * [N, 1, 3] -> [N, 8, 3]

    # 2. 绕z轴旋转边界框
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    ones, zeros = np.ones_like(rot_cos), np.zeros_like(rot_cos)
    rot_mat = np.array([[rot_cos, rot_sin, zeros],
                        [-rot_sin, rot_cos, zeros],
                        [zeros, zeros, ones]],  # 绕z轴旋转矩阵
                        dtype=np.float32)  # [3, 3, N]

    rot_mat = np.transpose(rot_mat, (2, 0, 1))  # [N, 3, 3]
    bboxes_corners = np.matmul(bboxes_corners, rot_mat)  # [N, 8, 3]

    # 3. 平移回原中心位置
    bboxes_corners += centers[:, None, :]
    return bboxes_corners

def project_3d_to_2d(all_corners, intrinsic, extrinsic):
    """
    将LiDAR坐标系的3D边界框角点投影到画面
    @param all_corners:     np.ndarray  [N, 8, 3]   所有边界框的8个3D角点
    @param intrinsic:       np.ndarray  [3, 3]      相机内参矩阵
    @param extrinsic:       np.ndarray  [3, 4]      相机外参矩阵
    @return:                np.ndarray  [N, 8, 2]   投影后的2D坐标
    """
    num_bbox = all_corners.shape[0]

    # 1. 扩展3D角点为齐次坐标: [N, 8, 3] -> [N, 8, 4]
    ones = np.ones((num_bbox, 8, 1))
    all_corners_homogeneous = np.concatenate([all_corners, ones], axis=-1)  # (X, Y, Z, 1)

    # 2. 将LiDAR坐标系下3D角点转换到相机坐标系: [N, 8, 4] -> [N, 8, 4]
    corners_3d_camera = np.matmul(all_corners_homogeneous, extrinsic.T)  # (X', Y', Z', 1)

    # 3. 将相机坐标系中的3D点投影到图像坐标系: [N, 8, 3] -> [N, 8, 3]
    corners_2d_homogeneous = np.matmul(corners_3d_camera[:, :, :3], intrinsic.T)  # (u, v, w)

    # 4. 将齐次坐标的归一化 (去除w分量): [N, 8, 3] -> [N, 8, 2]
    #    (u, v) = (u/w, v/w)
    corners_2d = corners_2d_homogeneous[:, :, :2] / corners_2d_homogeneous[:, :, 2:]  # (u, v)
    return corners_2d
