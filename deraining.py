# -*- coding = utf-8 -*-
# @Time : 2023/4/5 15:33
# @Author : DL
# @File : deraining.py
# @Software : PyCharm
import Dehazed.defog_v2 as defog
import Derain.PreNet_rtest as derain
import torch
import cv2
import numpy as np
from torchvision.utils import save_image
import utiles
from tqdm.auto import tqdm


def deraining(input_path='./test_Real_Internet', model_path='./model_Version6/model_best.ckpt',
              output_path='./results', temp_path='./temp/temp_img.jpg', THRESHOLD=100):
    print("正在去雨...")
    # 加载模型
    dataloader, net = derain.prepareModel(input_path, model_path)

    # 用于打印日志
    cnt = 0
    total = 0

    for input, label in tqdm(dataloader):
        cnt += 1
        input = input.to('cuda')  # 用cuda加速测试，也可以不用，不用cuda加速测试速度会很慢

        with torch.no_grad():
            output_image, _ = net(input)  # 输出的是张量

            # 将Tensor保存为jpg图像，方便后续去雾读取正确格式的图片
            save_image(output_image, temp_path)

        # 读取缓存中的图片
        output_image = cv2.imread(temp_path)
        test_lap = output_image

        # 去雾
        output_image = defog.deFogging(output_image)
        output_image = np.power(output_image, 1.03)  # 稍微提升图像整体曝光程度（去雾后图像可能整体偏暗）

        '''
        此处添加模糊识别算法
        '''
        test_lap_value = utiles.LaplacianValue(output_image) - utiles.LaplacianValue(test_lap)
        if test_lap_value <= THRESHOLD:
            output_image = test_lap

        '''
        此处添加滤波器代码
        '''
        output_image = cv2.GaussianBlur(output_image, (3, 3), 1)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义拉普拉斯算子
        output_image = cv2.filter2D(output_image, -1, kernel=kernel)  # 锐化

        utiles.save_img(output_image, output_path, cnt)

    print("去雨完成，图像保存至 ", output_path)
    print("")