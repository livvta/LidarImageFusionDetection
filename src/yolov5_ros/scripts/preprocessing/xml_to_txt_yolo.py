#  xml_to_txt_yolo.py
import xml.etree.ElementTree as ET
import os
import shutil
import random

xml_file_path = '/home/harris/detection_ws/src/yolov5_ros/scripts/dataset/kitti/labels/xml/'  # 检查和自己的xml文件夹名称是否一致
images_file_path = '/home/harris/detection_ws/src/yolov5_ros/scripts/dataset/kitti/images/image_2/'  # 检查和自己的图像文件夹名称是否一致
# 改成自己的类别名称
classes = ['Pedestrian', 'Car', 'Cyclist']
# 数据集划分比例，训练集80%，验证集10%，测试集10%
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1
# 此处不要改动，只是创一个临时文件夹
if not os.path.exists('temp_labels/'):
    os.makedirs('temp_labels/')
txt_file_path = 'temp_labels/'


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotations(image_name):
    in_file = open(xml_file_path + image_name + '.xml')
    out_file = open(txt_file_path + image_name + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
        #     continue
        if cls not in classes == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


total_xml = os.listdir(xml_file_path)
num_xml = len(total_xml)  # XML文件总数

for i in range(num_xml):
    name = total_xml[i][:-4]
    convert_annotations(name)


# *********************************************** #
#  parent folder
#  --data
#  ----images
#       ----train
#       ----val
#       ----test
#  ----labels
#       ----train
#       ----val
#       ----test
def create_dir():
    if not os.path.exists('images/'):
        os.makedirs('images/')
    if not os.path.exists('labels/'):
        os.makedirs('labels/')
    if not os.path.exists('images/train'):
        os.makedirs('images/train')
    if not os.path.exists('images/val'):
        os.makedirs('images/val')
    if not os.path.exists('images/test'):
        os.makedirs('images/test/')
    if not os.path.exists('labels/train'):
        os.makedirs('labels/train')
    if not os.path.exists('labels/val'):
        os.makedirs('labels/val')
    if not os.path.exists('labels/test'):
        os.makedirs('labels/test')

    return


# *********************************************** #
# 读取所有的txt文件
create_dir()
total_txt = os.listdir(txt_file_path)
num_txt = len(total_txt)
list_all_txt = range(num_txt)  # 范围 range(0, num)

num_train = int(num_txt * train_percent)
num_val = int(num_txt * val_percent)
num_test = num_txt - num_train - num_val

train = random.sample(list_all_txt, num_train)
# train从list_all_txt取出num_train个元素
# 所以list_all_txt列表只剩下了这些元素：val_test
val_test = [i for i in list_all_txt if not i in train]
# 再从val_test取出num_val个元素，val_test剩下的元素就是test
val = random.sample(val_test, num_val)
# 检查两个列表元素是否有重合的元素
# set_c = set(val_test) & set(val)
# list_c = list(set_c)
# print(list_c)
# print(len(list_c))

print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(train), len(val), len(val_test) - len(val)))
for i in list_all_txt:
    name = total_txt[i][:-4]

    srcImage = images_file_path + name + '.png'
    srcLabel = txt_file_path + name + '.txt'

    if i in train:
        dst_train_Image = 'images/train/' + name + '.png'
        dst_train_Label = 'labels/train/' + name + '.txt'
        shutil.copyfile(srcImage, dst_train_Image)
        shutil.copyfile(srcLabel, dst_train_Label)
    elif i in val:
        dst_val_Image = 'images/val/' + name + '.png'
        dst_val_Label = 'labels/val/' + name + '.txt'
        shutil.copyfile(srcImage, dst_val_Image)
        shutil.copyfile(srcLabel, dst_val_Label)
    else:
        dst_test_Image = 'images/test/' + name + '.png'
        dst_test_Label = 'labels/test/' + name + '.txt'
        shutil.copyfile(srcImage, dst_test_Image)
        shutil.copyfile(srcLabel, dst_test_Label)
print("complete")
