# -*- coding = utf-8 -*-
# @Time : 2023/4/5 15:33
# @Author : DL
# @File : defogging.py
# @Software : PyCharm

import Dehazed.defog_v2 as defog
import cv2
import utiles

def defogging(input_path='./realresults_PreNet_r_v6/0016.jpg', output_path='./defog_results'):
    print("正在去雾...")

    # 仅测试,一次只能处理一张
    input_img = cv2.imread(input_path)
    output_im = defog.deFogging(input_img)
    utiles.save_img(output_im, output_path, 0)

    print("去雾完成，图像保存至 ", output_path)
    print("")
