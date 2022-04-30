# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 5:33 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com

"""
UIUC翼型数据库加载类
References:https://blog.csdn.net/sinat_42239797/article/details/90641659
"""

from torch.utils.tensorboard import  SummaryWriter
import torch
from PIL import Image
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.FloatTensor)
import numpy  as np
from torchvision import  transforms
#
# class MyData(Dataset):
#     # 初始化文件路径以及文件列表
#     def __init__(self, img_dir, label_dir):
#
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         # 获取所在路径下所有图片的文件名
#         self.imgs = os.listdir(self.img_dir)
#         # 获取所在路径下所有标签的文件名
#         self.labels = os.listdir(self.label_dir)
#
#     # 返回图片以及对应标签名称
#     def __getitem__(self, item):
#         img_name = self.imgs[item]
#         print(img_name)
#         label_name = self.labels[item]
#         print(label_name)
#         img_item_path = os.path.join(self.img_dir, img_name)
#         label_item_path = os.path.join(self.label_dir, label_name)
#         img = Image.open(img_item_path)
#         torch_compose =transform.Compose([
#             # 将图像数据转换为张量的形式、
#             transform.ToTensor(),
#             transform.Resize((271, 271)),
#             # transform.Normalize(mean=(0.485), std=(0.229)),
#             # 统一图像大小
#             # 对图像进行正则化
#             # transform.Normalize(mean=(0.485, 0.456, 0.406, 0.406), std=(0.229, 0.224, 0.225, 0.225))
#         ])
#         img = torch_compose(img)
#         labels = np.array([])
#         file = open(label_item_path, 'r')
#         try:
#             for line in file:
#                 line = float(line)
#                 labels = np.append(labels, line)
#         finally:
#             file.close()
#         labels = torch.from_numpy(labels).float()
#         return img, labels
#
#     def __len__(self):
#         return len(self.imgs)

# 从标签中读取数据
def default_loader(imgs_path):
    return Image.open(imgs_path)

#**********************DataSet用户自定义数据加载类**********************
class MyData(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        super(MyData, self).__init__() # 对继承父类的属性进行初始化
        file = open(txt, 'r') # 读取包含图像路径以及标签的文本
        imgs = []
        for line in file:
            line = line.strip('\n') # 逐行读取，首先将当前行的换行符去掉
            words = line.split()  # 根据空格对字符串进行分割
            labels = words[1:]
            labels_float = np.array([])
            for i in range(len(labels)):
                 temp = float(labels[i])
                 labels_float = np.append(labels_float, temp)
            labels_float = torch.from_numpy(labels_float).float()
            imgs.append((words[0], labels_float))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

#**********************DataSet用户自定义数据加载类**********************


if __name__ == '__main__':
    root = "../data/CNN-FCN/labels_data/"
    trans = transforms.Compose([
        # 将图像数据转换为tensor，并且将像素归一化为0~1
        transforms.ToTensor(),
        # 将图片大小统一为216*216
        transforms.Resize((216, 216)),
        # transform.Normalize(mean=(0.485), std=(0.229)),
    ])
    # writer = SummaryWriter("log")
    train_data = MyData(txt=root + 'input_data.txt', transform=trans)
    # print(train_data[0])
    imgs, label = train_data[0]
    print(imgs)
    print(label)

    # imgs = torch.reshape(imgs,(1,1,216,216))
    # print(output_data.shape)
    # writer.add_images("test_my_data", imgs, global_step=1)
    # # imgs.show()
    # # print(imgs)
    # # print(output_data)
    # # print(imgs.shape)


