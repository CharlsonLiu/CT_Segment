"""
============================
# @Time    : 2022-08-09 22:38
# @Author  : HuangYJ
# @FileName: data.py
# @Software: PyCharm
# @desc    : 
===========================
"""
import os
import gol
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

# 记录当前处理的图片名称
present_name = ''


class MyDatasets(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png 标签名（图片名）
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)  # 标签地址
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))  # 原图地址
        # segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)

        mask = []
        mask.append(cv2.imread(os.path.join(segment_path), cv2.IMREAD_GRAYSCALE)[..., None])

        # 数组沿深度方向进行拼接。
        mask = np.dstack(mask)

        gol.set_value('present_name', segment_name)  # 设置一个全局变量
        return transform(image), transform(mask)


if __name__ == '__main__':
    data = MyDatasets('data')
    print(data[0][0].shape)
    print(data[0][1].shape)
