"""
============================
# @Time    : 2022-08-09 22:52
# @Author  : HuangYJ
# @FileName: utils.py
# @Software: PyCharm
# @desc    : 工具类
===========================
"""
from PIL import Image


def keep_image_size_open(path, size=(512, 512)):
    """
    防止图片resize后变形
    :param path: 图片地址
    :param size: 处理后大小
    :return: 处理后的图片
    """
    img = Image.open(path)
    maxLength = max(img.size)
    mask = Image.new('RGB', (maxLength, maxLength), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
