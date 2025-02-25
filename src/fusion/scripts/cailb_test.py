import sys
import numpy as np

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
    # P2 = np.array(list(map(float, raw[2].split()[1:]))).reshape((3, 4))
    P3 = np.array(list(map(float, raw[3].split()[1:]))).reshape((3, 4))
    R0 = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 3))
    R0 = np.hstack((R0, np.array([[0], [0], [0]])))
    R0 = np.vstack((R0, np.array([0, 0, 0, 1])))
    lidar2camera_m = np.array(list(map(float, raw[5].split()[1:]))).reshape((3, 4))
    lidar2camera_m = np.vstack((lidar2camera_m, np.array([0, 0, 0, 1])))
    imu2lidar_m = np.array(list(map(float, raw[6].split()[1:]))).reshape((3, 4))
    imu2lidar_m = np.vstack((imu2lidar_m, np.array([0, 0, 0, 1])))

    P2 = np.array(list(map(float, raw[7].split()[1:]))).reshape((3, 4))  # 此为Apollo摄像头内参
    extrinsic = np.array(list(map(float, raw[8].split()[1:]))).reshape((3, 4))  # 此为Apollo摄像头外参
    extrinsic = np.vstack((extrinsic, np.array([0, 0, 0, 1])))
    return P0, P1, P2, P3, R0, lidar2camera_m, imu2lidar_m, extrinsic


def calibration_param():
    """
    计算标定内外参数
    """
    P0, P1, P2, P3, R0, lidar2camera_matrix, imu2lidar_matrix, extrinsic= read_calib()
    intrinsic = P2[:, :3]  # Cam 2(color)
    # intrinsic = P2  # Cam 2(color)
    """
                    [fx   0  cx]
        intrinsic = [ 0  fy  cy]  fx, fy:相机焦距
                    [ 0   0   1]  cx, cy:相机主点
    """
    extrinsic = extrinsic
    # extrinsic = np.matmul(R0, lidar2camera_matrix)
    """
        extrinsic = [R | T]       R:旋转矩阵[3, 3]
                                  T:平移向量[3, 1]
    """
    # print("intrinsic\n", intrinsic)
    # print("extrinsic\n", extrinsic)

    return intrinsic, extrinsic


calibration_param()