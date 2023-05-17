# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         netmodel
# Description:
# Author:       huangmeng
# Date:         2023/5/17
# -------------------------------------------------------------------------------
# 模型构建
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5,stride=1) #卷积成1
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5,stride=1)#卷积成2
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=20*4*4, out_features=50) #经过卷积connv1和connv2之后shape为(1,20,4,4)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        # print("input输入大小:", x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) #
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # print("最后一层网络输出大小:",x.shape)
        print("输出层数据(没有经过进过函数log_softmax)",x)
        return F.log_softmax(x, dim=1)
