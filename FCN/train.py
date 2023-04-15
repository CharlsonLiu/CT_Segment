"""
============================
# @Time    : 2022-08-10 0:26
# @Author  : HuangYJ
# @FileName: train.py
# @Software: PyCharm
# @desc    : 
===========================
"""
import csv
import os.path
from glob import glob
from metrics import *
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader

from models.UNet_2Plus import UNet_2Plus
from models.UNet_3Plus import *
from data import *
from torchvision.utils import save_image
from models.FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 网络
vgg_model = VGGNet(requires_grad=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=1)
fcn_model = fcn_model.to(device)

criterion = nn.BCELoss().to(device)
# criterion = nn.BCEWithLogitsLoss().to(device)

optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.9)

weight_path = 'params'
best_weight_path = 'FCN/params/bestWeight/fcn_epoch_5.pth'
data_train_path = 'DATA-3/train'
data_val_path = 'DATA-3/val'

train_iou_list = []
val_iou_list = []
train_loss_list = []
val_loss_list = []

iteration = 60

plt.ion() # 开启交互模式
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
plt.title('Train & Val ==> IoU & Loss')
plt.xlabel('Epoch')
plt.ylabel('IoU')

# 设置坐标轴范围
ax.set_xlim(0, iteration-1)
ax.set_ylim(0, 1)

if __name__ == '__main__':
    gol._init()

    data_train_loader = DataLoader(MyDatasets(data_train_path), batch_size=1, shuffle=True)
    data_val_loader = DataLoader(MyDatasets(data_val_path), batch_size=1, shuffle=True)

    if os.path.exists(best_weight_path):
        fcn_model.load_state_dict(torch.load(best_weight_path))
        print('Loading weight successfully!')
    else:
        print('Fail to load weight!')

    # 创建csv文件
    csvfile = open("train_result.csv", "w+", newline='')
    writer = csv.writer(csvfile)
    writer.writerow(('epoch', 'train_loss', 'train_iou', 'val_loss', 'val_iou'))

    epoch = 1
    print('训练开始')
    while epoch <= iteration:
        train_loss = 0
        train_iou = 0
        val_loss = 0
        val_iou = 0

        print(f'<<<<<<<<<<<EPOCH: {epoch} >>>>>>>>>>>>>')

        # 对训练集：
        avg_meters_train = {'loss': AverageMeter(),
                            'iou': AverageMeter()}
        fcn_model.train()

        for i, (image, segment_img) in enumerate(data_train_loader):
            image, segment_img = image.to(device), segment_img.to(device)

            out_image = fcn_model(image)
            iou = iou_score(out_image, segment_img)

            out_image = torch.sigmoid(out_image)

            # 计算训练集损失和iou
            loss = criterion(out_image, segment_img)

            # 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters_train['loss'].update(loss.item(), image.size(0))
            avg_meters_train['iou'].update(iou, image.size(0))

            if i % 100 == 0:
                a = avg_meters_train['loss'].avg
                b = avg_meters_train['iou'].avg
                print(f'中间结果：epoch：{epoch}-{i}========》train_loss：{a}，train_iou：{b}')

            # weight_save_path = weight_path + '/fcn_epoch_' + str(epoch) + '.pth'
            # torch.save(fcn_model.state_dict(), weight_save_path)

        train_loss = avg_meters_train['loss'].avg
        train_iou = avg_meters_train['iou'].avg
        print(f'EPOCH：{epoch}========》train_loss：{train_loss}，train_iou：{train_iou}')

        # 对验证集
        avg_meters_val = {'loss': AverageMeter(),
                          'iou': AverageMeter()}
        # switch to evaluate mode
        fcn_model.eval()

        with torch.no_grad():
            for i, (image, segment_img) in enumerate(data_val_loader):
                image, segment_img = image.to(device), segment_img.to(device)

                out_image = fcn_model(image)
                iou = iou_score(out_image, segment_img)

                out_image = torch.sigmoid(out_image)

                # 计算验证集损失和iou
                loss = criterion(out_image, segment_img)


                avg_meters_val['loss'].update(loss.item(), image.size(0))
                avg_meters_val['iou'].update(iou, image.size(0))

            val_loss = avg_meters_val['loss'].avg
            val_iou = avg_meters_val['iou'].avg
            print(f'EPOCH：{epoch}========》val_loss：{val_loss}，val_iou：{val_iou}')

        # 写入csv
        writer.writerow((epoch, train_loss, train_iou, val_loss, val_iou))
        epoch += 1

        train_iou_list.append(train_iou)
        val_iou_list.append(val_iou)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        plt.plot(train_iou_list, 'b.-')  # 绘制训练集IoU折线图
        plt.plot(val_iou_list, 'g.-')  # 绘制验证集IoU折线图
        plt.plot(train_loss_list, 'r.-')  # 绘制训练集Loss折线图
        plt.plot(val_loss_list, 'k.-')  # 绘制验证集Loss折线图
        #plt.legend()
        plt.draw()  # 重绘
        plt.pause(0.05)  # 暂停0.05秒，使图像有更新的效果

    print('训练结束')
    csvfile.close()

    # 计算训练集和验证集iou的均值和方差
    train_iou_mean, train_iou_var = np.mean(train_iou_list), np.var(train_iou_list)
    val_iou_mean, val_iou_var = np.mean(val_iou_list), np.var(val_iou_list)

    # 绘制均值和方差到图像右下角
    plt.text(0.8, 0.05,
             f"Train IOU Mean: {train_iou_mean:.4f}\nTrain IOU VAR: {train_iou_var:.4f}\nVal IOU Mean: {val_iou_mean:.4f}\nVal IOU VAR: {val_iou_var:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
    plt.legend(['Train IoU', 'Val IoU', 'Train Loss', 'Val Loss'], loc='upper right')
    plt.show(block=True)

