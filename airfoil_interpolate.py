# -*- coding: utf-8 -*-
# @Time    : 2022/1/22 9:07 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com
"""
function✈:对UIUC翼型数据进行插值，保证数据的一致性
以NACA0010为基准翼型获取固定的70个X坐标值（基准翼型的选择具有随机性）
"""

from scipy import interpolate
import os
import glob
import numpy as np
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
import  matplotlib
matplotlib.rcParams['backend'] = 'SVG'
# 将全部的翼型数据点统一为70个
N = 70
k = 3

"""
function:用于将任意翼型进行任意数据点的插值
"""
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


"""
利用max-min分别对机翼的X、Y坐标进行标准化处理，范围为【0~1】
x - min/(max - min)
Single:对翼型的X以及Y轴分别做归一化，需要单独传入X或者Y坐标数值
Double:直接传入单个翼型的X、Y轴坐标数值，整体做归一化
"""

def Single_Normalization(data):
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
    x = Single_Normalization(x)
    y = Single_Normalization(y)

    for j in range(len(x)):
        m = []
        t1 = x[j]
        t2 = y[j]
        m.append(t1)
        m.append(t2)
        result.append(m)
    return result

# 获取固定的X坐标数值，并将其写入到指定文件中
def  get_fixed_X_axis(file):
    f = open(f'../data/CNN-FCN/X_axis_goe518', mode='w', encoding='utf8')
    for j in range(len(file)):
        print("写入文件的数据为:{}".format(file[j]))
        data = str(file[j]) + '\n'
        f.writelines(data)
        f.flush()
    f.close()

"""
function:单个翼型数据插值
"""
def airfoil_Normalization(file,name, N, k):
    # 对机翼数据进行插值到固定数量
    # 将翼型数据统一为70个翼型点坐标
    x_new, y_new, fp, ier = interpolates(file, N, k)
    # 将获得的新的翼型数据进行归一化
    # x_new = Single_Normalization(x_new)
    # y_new = Single_Normalization(y_new)
    x_root = file[:, 0]
    y_root = file[:, 1]

    dir = "../data/CNN-FCN/x_average.txt"
    x_averge = np.loadtxt(dir)

    # x_root = Single_Normalization(x_root)
    # y_root = Single_Normalization(y_root)
    # 获取拟合曲线，固定X坐标得到Y坐标
    # get_fixed_X_axis(x_new)

    plt.figure(figsize=(5.12, 5.12))
    # 原始翼型坐标
    plt.plot(x_root, y_root, 'k--', alpha=.5, label="True Value")
    # 插值后的翼型坐标
    plt.plot(x_new, y_new, 'mo', alpha=.5, label="Fit Value")
    # 固定X轴后得到的翼型坐标
    # plt.plot(x_averge, y_new, 'go', alpha=.5, output_data="input_data Value")
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc=2)  # 指定legend的位置,读者可以自己help它的用法
    plt.title('UIUC airfoil -- {}'.format(name))
    # plt.savefig('rae2822.pdf', format='pdf')
    plt.show()

if __name__ == '__main__':
    # dir = "../data/uiuc/naca0021.dat"   # 35 data points
    dir = "../data/uiuc/rae2822.dat"  # 129 data points
    # dir = "../data/uiuc/s2027.dat"
    name = 'rae2822'
    file = np.loadtxt(dir)
    # files = os.listdir(dir)
    # for i in range(len(files)):
    #     file = files[i]
    #     file = np.loadtxt(file)
    airfoil_Normalization(file, name=name, N=N, k=k)