# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         gui
# Description:
# Author:       huangmeng
# Date:         2023/4/26
# -------------------------------------------------------------------------------
""" for user to control"""
# https://www.bilibili.com/video/BV1kK4y1D7vP/?spm_id_from=333.880.my_history.page.click&vd_source=a577598d366ab0e1265dfb0c21fdcc3d
import sys
import pickle
import gzip
import torch
from tkinter import *
from tkinter import ttk
import tkinter.font as tf
# import tensorflow as tf
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras import models, layers
# from keras.optimizers import RMSprop
from PIL import Image, ImageDraw, ImageQt, ImageTk
import numpy as np
import cv2 as cv
from models.netmodel import Net
import torchvision

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
print(type(data))
(x_train, y_train), (x_test, y_test), _ = data

network = Net()
# 加载已经训练好的模型
network.load_state_dict(torch.load('model.pth'))
count = 0


class Controller(Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.createWidgets()

        self.startFlag = False
        self.handWriting = Image.new("RGB", (200, 200), (0, 0, 0))
        self.imgDraw = ImageDraw.Draw(self.handWriting)
        self.tf = tf.Font(family="微软雅黑", size=50)

    def createWidgets(self):
        label1 = Label(window, text="欢迎使用手写数字识别系统，下面是使用说明").pack(
            side=TOP)
        label2 = Label(window,
                       text="1. 用鼠标输入一个数字或点击“从mnist数据集中提取图片”").pack(
            side=TOP, anchor=W)
        label3 = Label(window, text="2.点击“清除”键重新进入。").pack(side=TOP, anchor=W)
        label4 = Label(window, text="3.保存按钮可以将您在画布上所写的内容保存为.jpg文件。").pack(
            side=TOP, anchor=W)
        label5 = Label(window, text="").pack(side=TOP, anchor=W)

        self.modeCombobox = ttk.Combobox(window, width=20)
        self.modeCombobox["values"] = ("1. 从Mnist提取图片", "2. 手工书写器")
        self.modeCombobox.current(0)
        self.modeCombobox.pack()

        writeFrame = LabelFrame(window, text="数据输入区")
        buttonFrame = Frame(window)
        resultFrame = LabelFrame(window, text="识别结果")
        writeFrame.place(x=35, y=170, width=200, height=200)
        buttonFrame.place(x=265, y=210, width=135, height=150)
        resultFrame.place(x=450, y=200, width=120, height=150)

        self.canvas = Canvas(writeFrame, bg="black", width=200, height=200)
        self.canvas.bind("<B1-Motion>", self.writing)
        self.canvas.bind("<ButtonRelease>", self.stop)
        self.canvas.pack(fill=BOTH, expand=True)

        clearButton = Button(buttonFrame, text="清除", command=self.clear)
        saveButton = Button(buttonFrame, text="保存", command=self.save)
        mnisButton = Button(buttonFrame, text="从Mnist提取图片", command=self.extract)
        mnisButton.pack(side=TOP, anchor=W, expand=YES, fill=X)
        clearButton.pack(side=TOP, anchor=W, fill=X)
        saveButton.pack(side=TOP, anchor=W, expand=YES, fill=X)

        self.resultCanvas = Canvas(resultFrame)
        self.resultCanvas.pack(fill=BOTH, expand=YES)

    def writing(self, event):
        self.modeCombobox.current(1)
        self.resultCanvas.delete("all")
        if not self.startFlag:
            self.startFlag = True
            self.x = event.x
            self.y = event.y
        self.canvas.create_line((self.x, self.y, event.x, event.y), width=10, fill="white")
        self.imgDraw.line((self.x, self.y, event.x, event.y), fill="white", width=19)
        self.x = event.x
        self.y = event.y
        self.imgArrOrigin = np.array(self.handWriting)
        self.imgArr = cv.resize(self.imgArrOrigin, (28, 28))  # interpolation?
        self.imgArr = cv.cvtColor(self.imgArr, cv.COLOR_BGR2GRAY)  # 图片转灰度图片
        # self.imgArr = self.imgArr.reshape((1, 28, 28, 1)).astype('float') / 255
        self.imgArr = self.imgArr.reshape((1, 28, 28)).astype('float32')
        # self.imgArr = self.imgArr/255.0
        # self.imgArr = self.imgArr.reshape(1,28,28)
        print(torch.from_numpy(self.imgArr))
        result = network(torch.from_numpy(self.imgArr))
        a, predict = torch.max(result.data, dim=1)
        label = predict.item()
        print(label)
        self.resultCanvas.create_text(60, 55, text=str(label), fill="red", font=self.tf)

    def stop(self, event):
        self.startFlag = False

    def clear(self):
        self.canvas.delete("all")
        self.handWriting = Image.new("RGB", (200, 200), (0, 0, 0))
        self.imgDraw = ImageDraw.Draw(self.handWriting)
        self.resultCanvas.delete("all")

    def extract(self):
        self.canvas.delete("all")
        self.resultCanvas.delete("all")
        self.modeCombobox.current(0)
        # ################
        randomInt = np.random.randint(0, 1000)
        self.mnistArray = x_test[randomInt]
        print("随机数", randomInt)
        print("x_test类型", type(x_test))
        # self.mnistArray = self.mnistArray*255.0
        print("初始获取数据",self.mnistArray)
        print("初始获取数据类型", type(self.mnistArray))
        print("初始获取数据大小", self.mnistArray.shape)

        mnistArrayBig = cv.resize(self.mnistArray.reshape(28, 28), (200, 200), interpolation=cv.INTER_LINEAR)
        print("mnistArrayBig:",type(mnistArrayBig))
        print("Big获取数据", mnistArrayBig)
        print("Big数据类型", type(mnistArrayBig))
        print("Big数据大小", mnistArrayBig.shape)
        tensodata = torch.from_numpy(mnistArrayBig)
        print("tensodata:", type(tensodata))
        # self.mnistImage = ImageTk.PhotoImage(Image.fromarray(mnistArrayBig))

        img=torchvision.transforms.ToPILImage()(tensodata)
        print("img:",type(img))
        self.mnistImage = ImageTk.PhotoImage(img)
        # self.mnistArray=self.mnistArray / 255.0
        # self.mnistArray=self.mnistArray.reshape(28, 28)
        print(type(self.mnistArray))
        print("======")
        # 数据类型为unit16(即(CV_16U)的图像可以保存为PNG、JPEG、TIFF格式文件。
        # 数据类型为float32的图像可以保存成PFM、TIFF、OpenEXR、和Radiance HDR格式文件。
        cv.imwrite("hm.TIFF", self.mnistArray.reshape(28, 28, 1))
        ###########################
        # ##########调试开始###########
        # img =Image.open('4.png')
        # self.mnistImage = ImageTk.PhotoImage(img)
        # ##########调试结束###########
        print("mnistImage:",type(self.mnistImage))
        #####测试图片显示问题######################
        # cv.imshow("test",mnistArrayBig*255)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        #####测试图片显示问题结束######################
        self.canvas.create_image(100,100,image=self.mnistImage)
        # self.canvas.pack()
        self.mnistArray = self.mnistArray.reshape((1, 28, 28)).astype('float32')
        print(self.mnistArray.shape)

        # self.mnistArray = self.mnistArray / 255.0
        # self.mnistArray = self.mnistArray.reshape(1, 28, 28)
        result = network(torch.from_numpy(self.mnistArray))
        # result = network(tensodata)
        # label = np.argmax(result, axis=1)
        a, predict = torch.max(result.data, dim=1)
        label = predict.item()
        print(label)
        self.resultCanvas.create_text(60, 55, text=str(label), fill="blue", font=self.tf)

    def save(self):
        self.imgArr = np.array(self.handWriting)
        self.imgArr = cv.resize(self.imgArr, (28, 28))
        global count
        cv.imwrite(str(count) + ".jpg", self.imgArr)
        count = count + 1
        cv.imshow(str(count) + ".jpg", self.imgArr)
        print("file saved")


window = Tk()
window.title("Digit Recognition System")
window.geometry("600x400")
controller = Controller(master=window)
window.mainloop()
