# -*- coding: utf-8 -*-
# @Time    : 2022/1/24 1:57 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com

"""
绘制所有的UIUC翼型图片并保存
"""

import numpy as np
import os
import matplotlib.pyplot as plt


# 批量绘制翼型图像并保存
def plot_multiFigure():
    path = "../data/test/normalize/"
    # 返回path指定文件夹包含的文件或者文件夹名字的列表
    fileDir = os.listdir(path=path)

    # 遍历每一个NACA翼型文件，分别绘制并保存翼型图像
    for i in range(len(fileDir)):
        print("加载并绘制第{- " + str(i) + " -}机翼", fileDir[i])
        # 加载单个翼型几何参数
        file_path = os.path.join(path, fileDir[i])
        x = np.loadtxt('../data/CNN-FCN/x_average.txt')
        y = np.loadtxt(file_path)
        # 绘制单个图像，自定义大小
        fig, ax = plt.subplots(1, 1, figsize=[2.16, 2.16])
        # 定义画图的数据、类别等
        ax.plot(x, y, color='k', lw=3)
        # 设置坐标的显示范围
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-1, 1])
        # 不显示坐标
        plt.axis('off')
        plt.show()
        # 获取当前翼型名称
        name = os.path.splitext(fileDir[i])[0]
        # name = fileDir[i].split('.')
        # 保存翼型图像
        fig.savefig('../data/test.txt/{}.png'.format(name), dpi=100)
        # 关闭当前画布，减小内存占用
        plt.close()


def plot_single_figure():
    x = np.loadtxt('../data/CNN-FCN/x_average.txt')
    y = np.loadtxt("../data/test/normalize/naca0012.txt")
    # 绘制单个图像，自定义大小
    fig, ax = plt.subplots(1, 1, figsize=[2.16, 2.16])
    ax.plot(x, y, color='k', lw=3)
    # plt.axis('off')
    # ax.plot(x, y, 'ro', alpha=.5, output_data="True Value")
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-1, 1])
    plt.show()

    # plt.figure()
    # plt.plot(x, y, 'ro-', alpha=.5, output_data="True Value")
    # plt.axis('equal')
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.legend(loc=2)  # 指定legend的位置,读者可以自己help它的用法
    # plt.show()


if __name__ == '__main__':
    plot_multiFigure()
    # plot_single_figure()
