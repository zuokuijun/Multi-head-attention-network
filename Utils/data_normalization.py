# -*- coding: utf-8 -*-
# @Time    : 2022/1/28 5:03 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com
import matplotlib.pyplot as plt
import  numpy as np
"""
需要理解归一化与标准化的区别
归一化（Normalization）：把数据转换到(0,1)或者(-1,1)区间的数据映射方式
标准化（Standardization）：把数据转换到均值为0，标准差为1的数据映射方式
"""

# 将数据归一化到0~1之间
def max_min_normalization(data):
    max = np.max(data, 0)
    min = np.min(data, 0)
    avl = max - min
    n_data = []
    for i in range(len(data)):
        temp = (data[i] - min) / avl
        n_data.append(temp)
    return n_data


def Double_Noremalization(data):
    x = []
    y = []
    result = []
    for i in range(len(data)):
        temp = data[i]
        x.append(temp[0])
        y.append(temp[1])
    x = max_min_normalization(x)
    y = max_min_normalization(y)

    for j in range(len(x)):
        m = []
        t1 = x[j]
        t2 = y[j]
        m.append(t1)
        m.append(t2)
        result.append(m)
    return result

# 将数据归一化到-1~1
def mean_normalization(data):
    # 获取
    max = np.max(data, 0)
    min = np.min(data, 0)
    avg = np.mean(data)
    result = []
    for i in range(len(data)):
        temp = (data[i] - avg)/(max - min)
        result.append(temp)
    return result

# 数据的反归一化
# x_new = x_new * (x_max - x_min) + x_avg
# path: 原始数据的路径
# data: 需要反归一化的数据
def  reverse_mean_normalization(path):
     # 加载原始数据获得当前翼型数据最大值、最小值以及平均值
     file  = np.loadtxt(path)
     max = np.max(file, 0)
     min = np.min(file, 0)
     average = np.mean(file)
     print("max:{}, min:{},average:{}".format(max,min,average))
     data = np.loadtxt("../CNN/naca0024_pre.txt")
     #---------------数据反归一化-----------------------------------
     reverse_data = []
     for i  in range(len(data)):
          temp = data[i] * (max - min) + average
          reverse_data.append(temp)
          # np.append(temp)
     #------------------------------------------------------------
     y = np.loadtxt("../data/CNN-FCN/y_coordinate/naca0024.txt")
     x =  np.loadtxt("../data/CNN-FCN/x_average.txt")
     plt.figure()
     plt.xlim(-0.02, 1.02)
     plt.ylim(-0.5, 0.5)
     plt.plot(x, reverse_data, '--', alpha=0.5, label="Prediction Value")
     plt.plot(x, y, 'ro', alpha=0.5, label="True Value")
     plt.legend(loc=1)  # 指定legend的位置,读者可以自己help它的用法
     # plt.title("UIUC airfoil --naca4421")
     plt.xlabel("x")
     plt.ylabel("y")
     plt.savefig("naca0024.png", dpi=1400)
     plt.show()

if __name__ == '__main__':
    path = "../data/CNN-FCN/y_coordinate/naca0024.txt"
    reverse_mean_normalization(path=path)
