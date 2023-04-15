# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : multi_network_IoU_weighting.py
# Time       ：2023/3/19 11:38
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import argparse
import csv
import json
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
import albumentations as A
import archs
import losses
from FCN import gol
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter

dir = 'inputs/DATA-3/test/JPEGImages'
# 统计输入文件夹中图片的数量
num_images = len(os.listdir(dir))

# 创建一个空的折线图
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot([], [], 'b.-', label='IoU')  # 用于绘制IoU值的折线图
ax.legend(loc='upper right')
plt.title('NestedUNet_wDS Intersection over Union')

# 设置坐标轴范围
ax.set_xlim(0, num_images)
ax.set_ylim(0, 1)


def parse_args():
    """
    需要指定参数：--name dsb2018_96_NestedUNet_woDS
    """
    parser = argparse.ArgumentParser()

    # 使用models下的文件名
    # ['DATA-3_UNet_woDS','DATA-3_NestedUNet_woDS', 'DATA-3_NestedUNet_wDS']
    parser.add_argument('--name', default="DATA-3_NestedUNet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    if config['arch'] == 'UNet':
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'], )
    elif config['arch'] == 'NestedUNet':
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'UNet_3Plus':
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'], )

    model = model.cuda()

    # Data loading code
    test_img_ids = glob(os.path.join('inputs', config['dataset'], 'test', 'JPEGImages', '*' + config['img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    val_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'test', 'JPEGImages'),
        mask_dir=os.path.join('inputs', config['dataset'], 'test', 'SegmentationClass'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    iou_list = []  # 新建一个list，存储每次测试的iou的值

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            loss = criterion(output, target)
            iou = iou_score(output, target)
            iou_list.append(iou.item())

            # 更新折线图数据
            line1.set_xdata(np.arange(len(iou_list)))
            line1.set_ydata(np.array(iou_list))

            # 重绘折线图
            fig.canvas.draw()

            # 在每次迭代中等待0.1秒
            plt.pause(0.1)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))
        with open('unet2p_iou_values_3.csv', mode='w') as csv_file:
            fieldnames = ['image_id', 'iou']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i, iou in enumerate(iou_list):
                writer.writerow({'image_id': i, 'iou': iou})

    mean_iou = np.mean(iou_list)
    var_iou = np.var(iou_list)
    # 添加均值和方差到文本框
    textstr = '\n'.join((
        r'$\mathrm{Mean\ IoU}=%.5f$' % (mean_iou,),
        r'$\mathrm{Var\ IoU}=%.5f$' % (var_iou,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.show(block=True)

    print('test_loss: ' + str(avg_meters['loss'].avg) + '   IoU: %.4f' % avg_meters['iou'].avg)

    # 保存测试结果数据
    result = {'arch': config['arch'],
              'loss': avg_meters['loss'].avg,
              'IoU': avg_meters['iou'].avg}
    result_path = 'outputs/' + config['name'] + '/test_result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f)

    # plot_examples(input, target, model, num_examples=3)

    torch.cuda.empty_cache()


def plot_examples(datax, datay, model, num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    print(datax.shape)
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx + 1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][0].set_title("Orignal Image" + str(image_indx))

        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0, :, :].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")

        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][2].set_title("Target image")

    plt.show()


if __name__ == '__main__':
    main()
