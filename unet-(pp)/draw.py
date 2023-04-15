"""
============================
# @Time    : 2022-08-13 19:58
# @Author  : HuangYJ
# @FileName: draw.py
# @Software: PyCharm
# @desc    : 绘制图像
===========================
"""
import matplotlib.pyplot as plt
import csv

uNetLog_path = 'models/DATA-3_UNet_woDS/log_2.csv'
uNetPlus_path = 'models/DATA-3_NestedUNet_woDS/log_2.csv'
uNetPlusWDS_path = 'models/DATA-3_NestedUNet_wDS/log_2.csv'


def draw(path):
    with open(path) as f:
        f_csv = csv.reader(f)

        epochs = []
        loss = []
        iou = []
        val_loss = []
        val_iou = []

        head = next(f_csv)
        i = 0
        for row in f_csv:
            i = i + 1
            epochs.append(i)
            loss.append(float(row[2]))
            iou.append(float(row[3]))
            val_loss.append(float(row[4]))
            val_iou.append(float(row[5]))

        plt.figure(figsize=(12, 8), dpi=80)  # 创建画布
        plt.grid(True, linestyle='--', alpha=0.5)  # 默认是True，风格设置为虚线，alpha为透明度
        plt.plot(epochs, iou, label='Training iou')
        plt.plot(epochs, val_iou, label='validation iou')
        plt.title('Training and validation iou')
        plt.legend()  # 显示图例
        if path == uNetLog_path:
            plt.savefig('graph/unet/Training and validation iou.png')
        elif path == uNetPlus_path:
            plt.savefig('graph/unet++/Training and validation iou.png')
        elif path == uNetPlusWDS_path:
            plt.savefig('graph/unet++WDS/Training and validation iou.png')

        plt.figure(figsize=(12, 8), dpi=80)  # 创建画布
        plt.grid(True, linestyle='--', alpha=0.5)  # 默认是True，风格设置为虚线，alpha为透明度
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()  # 显示图例
        if path == uNetLog_path:
            plt.savefig('graph/unet/Training and validation loss.png')
        elif path == uNetPlus_path:
            plt.savefig('graph/unet++/Training and validation loss.png')
        elif path == uNetPlusWDS_path:
            plt.savefig('graph/unet++WDS/Training and validation loss.png')
        plt.show()


if __name__ == '__main__':
    # uNetLog_path uNetPlus_path uNetPlusWDS_path
    draw(uNetPlusWDS_path)
