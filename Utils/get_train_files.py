# -*- coding: utf-8 -*-
# @Time    : 2022/1/24 8:44 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com

"""
function： 用于将待训练的翼型图像以及标签数据写入到一个txt文件中进行保存
Reference: https://blog.csdn.net/sinat_42239797/article/details/90641659
"""
import numpy as np
import os


def get_train_txt():
    # 对所有的训练标签数据进行处理，每条训练数据分别作为一行
    train_path = "../data/CNN-FCN/train_labels/"
    files = os.listdir(train_path)
    train_txt = open(f'../data/CNN-FCN/labels_data/train.txt', mode='w', encoding='utf8')
    # 分别对每个训练标签数据进行处理
    for i in range(len(files)):
        # 获取翼型坐标标签绝对路径
        airfoil_path = os.path.join(train_path, files[i])
        # 获取翼型名称
        airfoil_name = os.path.splitext(files[i])[0]
        # 加载翼型标签
        airfoil = np.loadtxt(airfoil_path)
        train_txt.write("../data/CNN-FCN/train_imgs/{}.png".format(airfoil_name) + " ")
        for j in range(len(airfoil)):
            print("写入文件的数据为:{}".format(airfoil[j]))
            data = str(airfoil[j]) + ' '
            train_txt.write(data)
            train_txt.flush()

        train_txt.write('\n')

    train_txt.close()


def get_test_txt():
    # 对所有的训练标签数据进行处理，每条训练数据分别作为一行
    test_path = "../data/CNN-FCN/val_labels/"
    files = os.listdir(test_path)
    test_txt = open(f'../data/CNN-FCN/labels_data/val.txt', mode='w', encoding='utf8')
    # 分别对每个训练标签数据进行处理
    for i in range(len(files)):
        # 获取翼型坐标标签绝对路径
        airfoil_path = os.path.join(test_path, files[i])
        # 获取翼型名称
        airfoil_name = os.path.splitext(files[i])[0]
        # 加载翼型标签
        airfoil = np.loadtxt(airfoil_path)
        test_txt.write("../data/CNN-FCN/val_imgs/{}.png".format(airfoil_name) + " ")
        for j in range(len(airfoil)):
            print("写入文件的数据为:{}".format(airfoil[j]))
            data = str(airfoil[j]) + ' '
            test_txt.write(data)
            test_txt.flush()

        test_txt.write('\n')

    test_txt.close()


if __name__ == '__main__':
    # get_train_txt()
    get_test_txt()
