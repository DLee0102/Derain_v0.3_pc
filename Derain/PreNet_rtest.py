# -*- coding = utf-8 -*-
# @Time : 2023/1/16 22:49
# @Author : DL
# @File : PreNet_rtest.py
# @Software : PyCharm

# 测试环境：Pycharm + pytorch + cuda
# 注意：需修改模型加载路径和测试数据读取与保存路径

from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 需要粘贴训练所用的模型架构
class PReNet_r(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list

def prepareData(input_path_):
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 测试图像的路径，这里不用自己写dataset，直接用现成的ImageFolder即可，因为测试时不需要获取标签
    test_set = ImageFolder(input_path_, transform=test_tfm)

    return test_set

def prepareModel(input_path_, model_path_):
    net = PReNet_r(use_GPU=True).to('cuda')     # 用cuda加速测试，也可以不用，不用cuda加速测试速度会很慢
    net.load_state_dict(torch.load(model_path_, map_location=torch.device('cuda')))      # 加载训练好的模型参数
    net.eval()      # 将模型切换到测试模式

    test_set = prepareData(input_path_)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    return dataloader, net

