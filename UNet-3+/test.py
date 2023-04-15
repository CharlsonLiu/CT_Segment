"""
============================
# @Time    : 2022-08-09 22:52
# @Author  : HuangYJ
# @FileName: test.py
# @Software: PyCharm
# @desc    : 测试网络
===========================
"""
import csv
import json
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

import gol

from metrics import iou_score
from models.UNet_3Plus import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader

loss_fun = nn.BCELoss()
net = UNet_3Plus().cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = 'params/bestWeight/unet3+_epoch_28.pth'
data_test_path = 'DATA-3/test'

if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('Loading weight successfully!')
else:
    print('Fail to load weight!')

dir = 'DATA-3/test/SegmentationClass'
# 统计输入文件夹中图片的数量
num_images = len(os.listdir(dir))

# 创建一个空的折线图
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot([], [], 'b.-', label='IoU')  # 用于绘制IoU值的折线图
ax.legend(loc='upper right')
plt.title('U-Net3+ Intersection over Union')

# 设置坐标轴范围
ax.set_xlim(0, num_images)
ax.set_ylim(0, 1)

if __name__ == '__main__':
    gol._init()

    data_test_loader = DataLoader(MyDatasets(data_test_path), batch_size=1, shuffle=False)
    name = os.listdir(os.path.join(data_test_path, 'SegmentationClass'))

    # 对测试集
    avg_meters_test = {'loss': AverageMeter(),
                       'iou': AverageMeter()}
    # switch to evaluate mode
    net.eval()

    iou_list = []#新建一个list，存储每次测试的iou的值

    with torch.no_grad():
        for i, (image, segment_img) in enumerate(data_test_loader):
            image, segment_img = image.to(device), segment_img.to(device)

            out_image = net(image)

            segment_name = gol.get_value('present_name')
            save_image(out_image, 'result/' + segment_name)

            # 计算验证集损失和iou
            loss = loss_fun(out_image, segment_img)
            iou = iou_score(out_image, segment_img)
            iou_list.append(iou.item())

            # 更新折线图数据
            line1.set_xdata(np.arange(len(iou_list)))
            line1.set_ydata(np.array(iou_list))

            # 重绘折线图
            fig.canvas.draw()

            # 在每次迭代中等待0.1秒
            plt.pause(0.1)

            print('result/' + segment_name + f'test: test_loss：{loss.item()}，test_iou：{iou}')

            avg_meters_test['loss'].update(loss.item(), image.size(0))
            avg_meters_test['iou'].update(iou, image.size(0))
        a = avg_meters_test['loss'].avg
        b = avg_meters_test['iou'].avg
        print(f'test: test_loss：{a}，test_iou：{b}')

        # 保存测试结果数据
        result = {'arch': 'UNet3+',
                  'loss': a,
                  'IoU': b}
        result_path = 'test_result.json'
        with open(result_path, 'w') as f:
            json.dump(result, f)
        with open('unet3p_iou_values.csv', mode='w') as csv_file:
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