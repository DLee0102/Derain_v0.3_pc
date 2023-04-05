# -*- coding = utf-8 -*-
# @Time : 2023/4/5 16:06
# @Author : DL
# @File : main.py
# @Software : PyCharm

import defogging as df
import deraining as dr
import srrebuilding as sr

if __name__ == "__main__":
    dr.deraining()  # 去雨（进度条效果由 tqdm 库实现）
    df.defogging()  # 去雾
    sr.srrebuilding()   # 超分辨重建（运行时间较长，大概一到两分钟）