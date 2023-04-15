# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : multi_network_IoU_weighting.py
# Time       ：2023/3/19 11:38
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import csv
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

#计算加权图与标签的iou
def weighted_iou(mask, weighted_mask):
    intersection = np.sum(np.minimum(mask, weighted_mask))
    union = np.sum(np.maximum(mask, weighted_mask))
    iou = intersection / union
    return iou

# 输入文件夹列表
dir_list = ['FCN/result', 'unet-(pp)/outputs/DATA-3_NestedUNet_wDS/0', 'UNet-3+/result','FCN/DATA-3/test/SegmentationClass']

# 输出文件夹
output_dir = 'multi_network_output'

#设定均值与标准差在评价时的权重
mean_weight = 7
var_weight = 3

# 读取iou数据
fcn_iou = pd.read_csv('FCN/fcn_iou_values.csv', header=0, usecols=[1], skiprows=[1]).values.flatten()
unet2p_iou = pd.read_csv('unet-(pp)/unet2p_iou_values_2.csv', header=0, usecols=[1], skiprows=[1]).values.flatten()
unet3p_iou = pd.read_csv('unet-(pp)/unet2p_iou_values_3.csv', header=0, usecols=[1], skiprows=[1]).values.flatten()

# 计算均值和标准差
fcn_mean, fcn_var = fcn_iou.mean(), fcn_iou.var()
unet2p_mean, unet2p_var = unet2p_iou.mean(), unet2p_iou.var()
unet3p_mean, unet3p_var = unet3p_iou.mean(), unet3p_iou.var()

fcn_var_weight = 1/fcn_var
unet2p_var_weight = 1/unet2p_var
unet3p_var_weight = 1/unet3p_var

#计算各标准差相应的权重
weights_sum = fcn_var_weight + unet2p_var_weight + unet3p_var_weight
fcn_var_weight /= weights_sum
unet2p_var_weight /= weights_sum
unet3p_var_weight /= weights_sum

#计算各算法IoU均值所占的权重
fcn_mean_weight = fcn_mean / (fcn_mean + unet2p_mean + unet3p_mean)
unet2p_mean_weight = unet2p_mean / (fcn_mean + unet2p_mean + unet3p_mean)
unet3p_mean_weight = unet3p_mean / (fcn_mean + unet2p_mean + unet3p_mean)

fcn_score = fcn_mean_weight*mean_weight + fcn_var_weight*var_weight
unet2p_score = unet2p_mean_weight*mean_weight + unet2p_var_weight*var_weight
unet3p_score = unet3p_mean_weight*mean_weight + unet3p_var_weight*var_weight

#归一化得到各自的权重
fcn_weight= fcn_score / (fcn_score + unet2p_score + unet3p_score)
unet2p_weight= unet2p_score / (fcn_score + unet2p_score + unet3p_score)
unet3p_weight= unet3p_score / (fcn_score + unet2p_score + unet3p_score)

print(f"FCN weight: {fcn_weight:.3f}")
print(f"UNet-2+ weight: {unet2p_weight:.3f}")
print(f"UNet-3+ weight: {unet3p_weight:.3f}")

# 读取输入文件夹的第一张图片，获取编号起始数字
first_img = cv2.imread(os.path.join(dir_list[0], os.listdir(dir_list[0])[0]), cv2.IMREAD_GRAYSCALE)
start_num = int(os.listdir(dir_list[0])[0].split('.')[0])

# 统计输入文件夹中图片的数量
num_images = len(os.listdir(dir_list[0]))

# 创建一个空的折线图
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot([], [], 'b.-', label='IoU')  # 用于绘制IoU值的折线图
ax.legend(loc='upper right')
plt.title('Weighted Intersection over Union')

# 设置坐标轴范围
ax.set_xlim(0, num_images)
ax.set_ylim(0, 1)

#根据各自的iou自动计算其权重
# weights = np.array([fcn_weight, unet2p_weight, unet3p_weight])
weights = np.array([1.5, 4.25, 1.25]) # 加权系数,可以自己调节

iou_list = []

for i in range(num_images):
    filename = f"{i+start_num}.png"
    output_img = np.zeros((512, 512), dtype=np.float32)
    img_list = []
    for dir in dir_list:
        img_path = os.path.join(dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = img.astype(np.float32) / 255.0 # 将像素值数据类型转换为浮点型，并将像素值范围归一化到[0, 1]
        else:
            print(f"Error: failed to read image {img_path}")
            continue
        img_list.append(img)
    img_array = np.array(img_list)

    label_path = os.path.join(dir_list[3], filename)
    label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)

    # 计算加权输出图像
    total_iou = np.zeros(fcn_iou.shape[0])
    total_iou[i] = weights[0]*fcn_iou[i] + weights[1]*unet2p_iou[i] + weights[2]*unet3p_iou[i]
    output_img = weights[0]*fcn_iou[i]/total_iou[i]*img_list[0]+\
                 weights[1]*unet2p_iou[i]/total_iou[i]*img_list[1]+\
                 weights[2]*unet3p_iou[i]/total_iou[i]*img_list[2]
    output_img = 255*output_img;
    output_img = output_img.astype(np.uint8)

    # 高斯滤波，平滑图片
    img = cv2.GaussianBlur(output_img, (3, 3), 0)
    # Canny算子，进行边缘检测
    edges = cv2.Canny(img, 100, 200)

    output_img = edges + output_img

    # 得到输出图片的尺寸
    height, width = output_img.shape

    #判断像素是否为主动脉血管的一部分的阈值
    threshold = 80

    for y in range(height):
        for x in range(width):
            # 访问像素 (x, y)
            pixel = output_img[y, x]
            if pixel >= threshold:
                output_img[y, x] = 255
            else:
                output_img[y, x] = 0

    #计算加权图片与原来标签图的iou
    iou = weighted_iou(label , output_img)
    iou_list.append(iou)

    # 保存加权输出图像
    output_dir = f"weighted_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f"{i+start_num}.png")
    print(f"Saving {output_filename}...")
    cv2.imwrite(output_filename, output_img)

    # 更新折线图数据
    line1.set_xdata(np.arange(len(iou_list)))
    line1.set_ydata(np.array(iou_list))

    # 重绘折线图
    fig.canvas.draw()

    # 在每次迭代中等待0.1秒
    plt.pause(0.05)

    with open('weighted_iou_values_2.csv', mode='w',newline='') as csv_file:
        fieldnames = ['image_id', 'iou']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, iou in enumerate(iou_list):
            writer.writerow({'image_id': i, 'iou': iou})

mean_iou = np.mean(iou_list)
var_iou = np.var(iou_list)

# 添加均值和方差到文本框
textstr = '\n'.join((
    r'$\mathrm{Mean\ IoU}=%.5f$' % (mean_iou, ),
    r'$\mathrm{Var\ IoU}=%.5f$' % (var_iou, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=14,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.show(block = True)