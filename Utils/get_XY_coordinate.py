# -*- coding: utf-8 -*-
# @Time    : 2022/1/23 6:37 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com

"""
获取翼型拟合后的X以及Y坐标
这里将拟合数据点设置为70个
"""

import os
import numpy as np
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
from Utils.data_normalization import mean_normalization


# 将全部的翼型数据点统一为70个
N = 70
k = 3

def interpolates(Q, N, k, D=20, resolution=1000):
    ''' Interpolate N points whose concentration is based on curvature. '''
    res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)

    tck, u = res
    #
    uu = np.linspace(u.min(), u.max(), resolution)
    x, y = splev(uu, tck, der=0)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    # 返回数字的绝对值
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D
    #
    cv_int = cumtrapz(cv, uu, initial=0)
    fcv = interp1d(cv_int, uu)
    cv_int_samples = np.linspace(0, cv_int.max(), N)
    u_new = fcv(cv_int_samples)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new, fp, ier


# 将插值后的X坐标进行保存
def save_x_coordinate(x_new, name):
    f = open(f'../data/test/x/{name}.txt', mode='w', encoding='utf8')
    for j in range(len(x_new)):
        print("写入文件的数据为:{}".format(x_new[j]))
        data = str(x_new[j]) + '\n'
        f.writelines(data)
        f.flush()
    f.close()


# 将插值后的Y坐标进行保存
def save_y_coordinate(y_new, name):
    f = open(f'../data/test/y/{name}.txt', mode='w', encoding='utf8')
    for j in range(len(y_new)):
        print("写入文件的数据为:{}".format(y_new[j]))
        data = str(y_new[j]) + '\n'
        f.writelines(data)
        f.flush()
    f.close()

# 将归一化后的Y坐标数据进行保存
def save_y_coordinate_normalization(y_new, name):
    f = open(f'../data/test/normalize/{name}.txt', mode='w', encoding='utf8')
    for j in range(len(y_new)):
        print("写入文件的数据为:{}".format(y_new[j]))
        data = str(y_new[j]) + '\n'
        f.writelines(data)
        f.flush()
    f.close()


def get_xy_coordinate(path):
    files = os.listdir(path)
    print("UIUC 翼型数据库总计翼型数量为{}".format(len(files)))
    for i in range(len(files)):
        print("正在导出第{}个翼型的XY坐标数据... ...".format(i+1))
        # 获取文件的绝对路径
        file_path = os.path.join(path, files[i])
        # 获取翼型文件名称
        file_name = os.path.splitext(files[i])[0]
        print("导出翼型{}的坐标数据... ...".format(file_name))
        file = np.loadtxt(file_path)
        x_new, y_new, fp, ier = interpolates(file, N=N, k=k)
        # 导出插值后的X坐标数据
        save_x_coordinate(x_new, file_name)
        # 导出插值后的Y坐标数据
        save_y_coordinate(y_new, file_name)
        # 导出插值且归一化后的Y坐标数据
        y_new_nor = mean_normalization(y_new)
        save_y_coordinate_normalization(y_new_nor, file_name)


def test_plot_xy():
    # x = "../data/CNN-FCN/x_coordinate/e378.txt"
    # x = "../data/CNN-FCN/x_average.txt"
    # y = "../data/CNN-FCN/y_coordinate/naca0006.txt"
    # y_1 = "../data/CNN-FCN/y_coordinate_normalization/naca0006.txt"
    y = "../data/uiuc/naca0012.dat"
    # x = np.loadtxt(x)
    m = np.loadtxt(y)
    print(type(m))
    print(m.shape)
    y1 = m[:, 0]
    y2 = m[:, 1]
    # y_1 = np.loadtxt(y_1)
    fig, ax = plt.subplots()
    plt.plot(y1, y2, 'bo-', alpha=.5, label="True Value")
    # plt.plot(x, y_1, 'ro-', alpha=.5, label="Normalization Value")
    plt.legend(loc=1)  # 指定legend的位置,读者可以自己help它的用法
    # ax.set_xlim([-0.1, 1.1])
    # ax.set_ylim([-0.5, 0.5])
    # ax.set_xlim([-0.05, 1.02])
    # ax.set_ylim([-0.3, 0.4])
    plt.title("UIUC airfoil --naca0012")
    plt.show()
    # plt.legend(loc=2)  # 指定legend的位置,读者可以自己help它的用法


if __name__ == '__main__':
    dir = "../data/naca0012/"
    get_xy_coordinate(dir)
    # test_plot_xy()



