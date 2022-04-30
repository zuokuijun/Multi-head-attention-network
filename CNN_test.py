# -*- coding: utf-8 -*-
# @Time    : 2022/1/13 9:03 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com

import torch
import torchvision.transforms
from PIL import Image
from cnn import CNN

model = CNN()
model.load_state_dict(torch.load("./CNN4900.pth", map_location=torch.device('cpu')))
# model = torch.load("CNN_FCN_Adam1000.pth", map_location=torch.device('cpu'))

image = Image.open("../data/CNN-FCN/test_imgs/naca0024.png", 'r')
# image = image.resize((216,216))
# print(image)
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),]
)
image = transform(image)
image = image.reshape(1, 1, 216, 216)

model.eval()
with torch.no_grad():
    output = model(image)
    print(output)
output = output.detach().numpy()

output = output[0]
f = open('naca1412_pre.txt', mode='w', encoding='utf8')

print(type(output))


for i in range(len(output)):
    data = output[i]
    print("写入测试的数据{}".format(data))
    f.writelines(str(data)+'\n')
    f.flush()

f.close()
