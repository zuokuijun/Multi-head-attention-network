# -*- coding: utf-8 -*-
# @Time    : 2022/1/21 9:32 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com
import matplotlib.pyplot as plt
import numpy as np
from numpy import polyfit, poly1d
from scipy import interpolate
import os
import glob
import numpy as np
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt

"""
用于对UIUC翼型数据库中的翼型数据进行拟合
然后在拟合的曲线上选择任意翼型数据点
拟合曲线X坐标数值重复的解决方法：https://fixexception.com/scipy/expect-x-to-not-have-duplicates/
"""
# 利用max-min分别对机翼的X、Y坐标进行标准化处理，范围为【0~1】
# x - min/(max - min)
def  Normalization(data):
     max = np.max(data, 0)
     min = np.min(data, 0)
     avl = max - min
     n_data = []
     for i in range(len(data)):
         temp = (data[i] - min)/avl
         n_data.append(temp)
     return n_data

"""
Q:待插值的二维翼型坐标
N：插值后的翼型坐标点
k:B样条插值超参数，一般设置为3
"""
def interpolate(Q, N, k, D=20, resolution=1000):
    ''' Interpolate N points whose concentration is based on curvature. '''
    res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)
    tck, u = res
    uu = np.linspace(u.min(), u.max(), resolution)
    x, y = splev(uu, tck, der=0)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D
    cv_int = cumtrapz(cv, uu, initial=0)
    fcv = interp1d(cv_int, uu)
    cv_int_samples = np.linspace(0, cv_int.max(), N)
    u_new = fcv(cv_int_samples)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new, fp, ier

# 输入需要拟合的X Y翼型坐标
def  uiuc_fit(file_path):
      foil = np.loadtxt(file_path)
      x = []
      y = []
      for i in range(len(foil)):
          temp = foil[i]
          x.append(temp[0])
          y.append(temp[1])

      x = Normalization(x)
      y = Normalization(y)
      print(x)
      print(y)
      # 上翼面坐标
      up_x = []
      up_y = []
      # 下翼面坐标
      down_x = x.copy()
      down_y = y.copy()

      if  len(x) % 2 == 0:
          print("翼型坐标为偶数")
      elif len(x) % 2 !=0:
          print("翼型坐标为奇数")
          m = int(len(x)/2)
          for i in range(m+1):
              up_x.append(x[i])
              down_x.remove(x[i])
          for j in range(m+1):
              up_y.append(y[j])
              down_y.remove(y[j])
      print(up_x)
      print(down_x)
      print(up_y)
      print(down_y)
      f1 = interpolate.interp1d(up_x, up_y, kind='quadratic')
      pre_up_y = f1(up_x)
      f2 = interpolate.interp1d(down_x,down_y,kind='quadratic')
      pre_down_y = f2(down_x)
      plot1 = plt.plot(up_x, up_y, 'o', color='b', label='original values')
      plot2 = plt.plot(up_x, pre_up_y, '-', color='r', label='polyfit values')
      plot3 = plt.plot(down_x, down_y, 'o', color='b', label='polyfit values')
      plot2 = plt.plot(down_x, pre_down_y, '-', color='r', label='polyfit values')
      # plot2 = plt.plot(down_x, down_y, '-', color='r', output_data='polyfit values')
      plt.xlabel('x axis')
      plt.ylabel('y axis')
      plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
      plt.title('polyfitting')
      plt.show()






if __name__ == '__main__':
    dir = "../data/uiuc/naca0006.dat"
    uiuc_fit(dir)

