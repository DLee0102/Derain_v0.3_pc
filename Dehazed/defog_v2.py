import cv2
import numpy as np

# 导向滤波（待研究）
def guided_filter(I,p,win_size,eps):

    mean_I = cv2.blur(I,(win_size,win_size))
    mean_p = cv2.blur(p,(win_size,win_size))

    corr_I = cv2.blur(I*I,(win_size,win_size))
    corr_Ip = cv2.blur(I*p,(win_size,win_size))

    var_I = corr_I-mean_I*mean_I
    cov_Ip = corr_Ip - mean_I*mean_p

    a = cov_Ip/(var_I+eps)
    b = mean_p-a*mean_I

    mean_a = cv2.blur(a,(win_size,win_size))
    mean_b = cv2.blur(b,(win_size,win_size))

    q = mean_a*I + mean_b
    return q

def get_min_channel(img):
    return np.min(img,axis=2)

def min_filter(img,r):
    kernel = np.ones((2*r-1,2*r-1))
    return cv2.erode(img,kernel)    # 最小值滤波器，可用腐蚀替代

def get_A(img_haze,dark_channel,bins_l):

    hist,bins = np.histogram(dark_channel,bins=bins_l)  # 得到直方图
    d = np.cumsum(hist)/float(dark_channel.size)    # 累加
    threshold=0

    # 设定一个阈值（220，归一化后的值约为0.86）
    MAX_A = 0.86
    for i in range(bins_l-1,0,-1):
        if d[i]<=0.999:
            threshold=i
            break
    A = img_haze[dark_channel>=bins[threshold]].mean()

    # 设定一个阈值（220，归一化后的值约为0.86），防止全局大气光值过大（255）
    if A >= MAX_A:
        A = MAX_A
    # 候选区域可视化
    show  = np.copy(img_haze)
    show[dark_channel>=bins[threshold]] = 0,0,255
    return A

def get_t(img_haze,A,t0=0.1,w=0.95):
    out = get_min_channel(img_haze)
    out = min_filter(out,r=7)
    t = 1-w*out/A   # 需要乘上一系数w，为远处的物体保留少量的雾
    t = np.clip(t,t0,1) # 论文提到t(x)趋于0容易产生噪声，所以设置一最小值0.1
    return t

def deFogging(input_img):
    I = input_img/255.0
    dark_channel = get_min_channel(I)
    dark_channel_1 = min_filter(dark_channel,r=7)

    A = get_A(I,dark_channel_1,bins_l=2000)

    t = get_t(I,A)
    t = guided_filter(dark_channel,t,81,0.001)
    t = t[:,:,np.newaxis].repeat(3,axis=2)  # 升维至(r,w,3)

    J = (I-A)/t +A

    J = np.clip(J,0,1)
    J = J*255
    J =np.uint8(J)

    output_img = J

    return output_img


