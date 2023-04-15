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

uNet3Plus_path = 'train_result_2.csv'


def draw(path):
    with open(path) as f:
        f_csv = csv.reader(f)

        epochs = []
        loss = []
        iou = []
        val_loss = []
        val_iou = []

        head = next(f_csv)
        print(head)
        i = 0
        for row in f_csv:
            if not row:
                continue
            i = i + 1
            epochs.append(i)
            loss.append(float(row[1]))
            iou.append(float(row[2]))
            val_loss.append(float(row[3]))
            val_iou.append(float(row[4]))

        plt.figure(figsize=(12, 8), dpi=80)  # 创建画布
        plt.grid(True, linestyle='--', alpha=0.5)  # 默认是True，风格设置为虚线，alpha为透明度
        plt.plot(epochs, iou, label='Training iou')
        plt.plot(epochs, val_iou, label='validation iou')
        plt.title('Training and validation iou')
        plt.legend()  # 显示图例
        plt.savefig('graph/Training and validation iou.png')

        plt.figure(figsize=(12, 8), dpi=80)  # 创建画布
        plt.grid(True, linestyle='--', alpha=0.5)  # 默认是True，风格设置为虚线，alpha为透明度
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()  # 显示图例
        plt.savefig('graph/Training and validation loss.png')

        plt.show()


if __name__ == '__main__':
    draw(uNet3Plus_path)
