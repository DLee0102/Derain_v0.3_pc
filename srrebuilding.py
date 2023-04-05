# -*- coding = utf-8 -*-
# @Time : 2023/4/5 15:33
# @Author : DL
# @File : srrebuilding.py
# @Software : PyCharm
from SR.utils import *
import time
from SR.srr_test import prepareModel, prepareData

def srrebuilding(input_path='D:/pythonProject1/Derain_v0.3_pc/srrebuild_test/0016.jpg',
                 sr_middle_result_path='./srr_temp',
                 srresnet_checkpoint="./SR/model/savecheckpoint_srresnet3.pth"):
    print("正在超分辨率重建...")
    start = time.time()

    lr_img = prepareData(input_path, sr_middle_result_path)
    model = prepareModel(srresnet_checkpoint=srresnet_checkpoint)
    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save(sr_middle_result_path + '/test_srres.jpg')

    print('用时  {:.3f} 秒'.format(time.time() - start))
    print("去雨完成，图像保存至 ", sr_middle_result_path)
    print("")
