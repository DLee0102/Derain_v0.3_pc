# -*- coding = utf-8 -*-
# @Time : 2023/4/5 16:17
# @Author : DL
# @File : srr_test.py
# @Software : PyCharm
# 此处若有报错提示 未解析的引用 'SR' 则不用管
from SR.utils import *
from SR.models import SRResNet
from PIL import Image


def prepareData(input_path_, sr_middle_result_path='./srr_temp'):
    # 加载图像
    img = Image.open(input_path_, mode='r')
    img = img.convert('RGB')

    # 双线性上采样
    Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)),
                             Image.Resampling.BICUBIC)
    # Bicubic_img.save(sr_middle_result_path + '/test_bicubic.jpg')

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
    return lr_img


# ssr模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 4  # 放大比例
device = torch.device("cpu")


def prepareModel(large_kernel_size=9, small_kernel_size=3, n_channels=64,
                 n_blocks=16, scaling_factor=4, device=torch.device("cpu"),
                 srresnet_checkpoint="./SR/model/savecheckpoint_srresnet3.pth"):
    # 预训练模型
    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srresnet_checkpoint, map_location=torch.device('cpu'))
    srresnet = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    srresnet = srresnet.to(device)
    srresnet.load_state_dict(checkpoint['model'])

    srresnet.eval()
    model = srresnet

    return model
