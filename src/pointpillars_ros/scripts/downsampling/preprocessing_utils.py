import numpy as np


def read_bin_file(bin_file_path):
    """
    读取bin文件
    @param bin_file_path: bin文件路径
    @return:    np.array [N, 4] N为点云数量, 4为点云数据格式
    """
    return np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)


def write_bin_file(bin_file_path, points):
    """
    写入bin文件
    @param bin_file_path: 输出路径
    @param points: np.array [N, 4] 下采样后的点云数据
    """
    points.tofile(bin_file_path)