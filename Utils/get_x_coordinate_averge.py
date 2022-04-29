# -*- coding: utf-8 -*-
# @Time    : 2022/1/23 6:33 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com
"""
1、将所有的翼型全部转换为相同的数据点
Reference: get_XY_coordinate.py
2、选取固定的X坐标点
 将数据库中所有的X坐标进行平均
3、对拟合后的全部翼型的X坐标取平均值，作为最终拟合的翼型X坐标
"""
import os

import numpy as np


def get_x_average(path):
    # 平均数值存储数组
    x_average = []
    x_files = os.listdir(path)
    # 总计翼型坐标数据个数
    num = len(x_files)
    # 总计需要平均的数据点
    for i in range(70):
        sum = 0
        # 对所有的文件进行取值
        for j in range(num):
            x_file_path = os.path.join(path, x_files[j])
            print(x_file_path)
            x = np.loadtxt(x_file_path)
            sum = sum + x[i]
        x_average.append(sum/num)

    print(x_average)
    f = open(f'../data/CNN-FCN/x_average.txt', mode='w', encoding='utf8')
    for j in range(len(x_average)):
        print("写入文件的数据为:{}".format(x_average[j]))
        data = str(x_average[j]) + '\n'
        f.writelines(data)
        f.flush()
    f.close()



if __name__ == '__main__':
    path = "../data/CNN-FCN/x_coordinate/"
    get_x_average(path)


