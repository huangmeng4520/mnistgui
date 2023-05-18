# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         model
# Description:
# Author:       huangmeng
# Date:         2023/5/17
# -------------------------------------------------------------------------------
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import netmodel

n_epochs = 3  # 循环整个训练数据集的次数
batch_size_train = 64  # 批大小
batch_size_test = 1000
learning_rate = 0.01  # 优化器参数
momentum = 0.5  # 优化器参数
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
# 数据训练集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/', train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       mean=(0.1307,), std=(0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
# print(type(train_loader))
#shuffle=True,在每个epoch开始的时候，对数据进行重新打乱
# 测试数据集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       mean=(0.1307,), std=(0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


def examples():
    print("train_loader类型",type(train_loader))
    examples = enumerate(train_loader)
    print("enumerate",type(examples))
    # for index,value in examples:
    #     print(index)
    #     print(type(value))
    #     print(len(value))
    #     print(value[0].shape)#64批次训练数据矩阵
    #     print(value[0][0][0])
    #     print(value[1].shape)#64批次图片对应的结果集
    batch_idx, (example_data, example_targets) = next(examples)
    # (example_data, example_targets) = list(examples)[0]
    print(example_data)
    print(example_data.shape)
    print(example_data.numpy().shape)
    print(example_targets)
    print(example_data.shape)
    # 数据集第一个图片数据
    transforms.ToPILImage()(example_data[63][0]).save('./test.png')
    print(example_data[0][0])
    print(example_targets[0])

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('./Ground Truth.png')
    plt.show()





network = netmodel.Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network.to(device)
# network.cuda()  # network.cuda()将网络参数发送给GPU
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):  # 获取训练数据集
        optimizer.zero_grad()  # 梯度设置为0

        output = network(data.to(device))  # 进行向前计算
        loss = F.nll_loss(output, target.to(device))  # 带权损失
        loss.backward()  # 进行反向传播计算梯度
        optimizer.step()  # 参数更新
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), '../model.pth')  # 保存模型参数
            torch.save(optimizer.state_dict(), '../optimizer.pth')  # 保存优化器参数


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(device))
            print(output)
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# https://zhuanlan.zhihu.com/p/137571225
# https://blog.csdn.net/NikkiElwin/article/details/112980305
# https://www.bilibili.com/video/BV1ir4y1U7q2?p=4&vd_source=a577598d366ab0e1265dfb0c21fdcc3d
def train_data(count=6):
    train(1)

    test()  # 不加这个，后面画图就会报错：x and y must be the same size
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = network(example_data.to(device))
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('./Prediction.png')
    plt.show()

    # ----------------------------------------------------------- #

    continued_network = netmodel.Net()
    continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    network_state_dict = torch.load('../model.pth')
    continued_network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load('../optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)

    # 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
    # 不然报错：x and y must be the same size
    # 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
    for i in range(4, count):
        test_counter.append(i * len(train_loader.dataset))
        train(i)
        test()

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('./Train or Test Loss.png')
    plt.show()


if __name__ == '__main__':
    # examples()
    train_data(100)
