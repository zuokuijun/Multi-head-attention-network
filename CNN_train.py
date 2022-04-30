# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 4:31 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com
import torch


from cnn import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from CNN.Data import MyData


################################################################
batchSize = 64           # 每次加载的数据量
epochs = 5000             # 数据变量次数
learningRate = 0.00025   # 初始化学习率
train_step = 0
test_step = 0
root = "../data/CNN-FCN/labels_data/" # 翼型数据标签文件路径
################################################################

#************************定义新的数据加载方式**************************
# trans = transforms.Compose([
#     # 将图像数据转换为tensor，并且将像素归一化为0~1
#     transforms.ToTensor(),
#     # # 将图片大小统一为216*216
#     # transforms.Resize((216, 216)),
#     # # transform.Normalize(mean=(0.485), std=(0.229)),
# ])
train_data = MyData(txt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyData(txt=root + 'val.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batchSize, shuffle=True)
#************************定义新的数据加载方式**************************

cnn = CNN().cuda()
loss = nn.MSELoss().cuda()
writer = SummaryWriter("flowField_log")
optim = torch.optim.Adam(cnn.parameters(), lr=learningRate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.1, last_epoch=-1)
for epoch in range(epochs):
    print("---------------------开始第{}轮网络训练-----------------".format(epoch + 1))
    # 训练神经网络
    train_loss = 0
    train_number = 0
    for data in train_loader:
        input, target = data
        input = input.cuda()
        target = target.cuda()
        output = cnn(input)
        target_loss = loss(output, target)
        optim.zero_grad()
        target_loss.backward()
        optim.step()
        train_step = train_step + 1
        train_number += 1  # 统计总计有多少个batchsize
        train_loss += target_loss.item()  # 统计所有batchsize 的总体损失
        if train_step % 100 == 99:
            print("神经网络训练次数[{}], Loss:{}".format(train_step, target_loss.item()))
        writer.add_scalar("train_loss", target_loss.item(), global_step=train_step)
    # writer.add_scalar("epoch_train_loss", train_loss, epoch + 1)  # 当前批次的训练总体损失
    # writer.add_scalar("epoch_train_number", train_number, epoch + 1)  # 当前批次的训练总数
    writer.add_scalar("epoch_train_avg_loss", train_loss / train_number, epoch + 1)  # 当前批次训练的平均损失
    lr_scheduler.step()
    # 测试神经网络
    test_total_loss = 0
    test_number = 0
    with torch.no_grad():
        for data in test_loader:
            input, target = data
            input = input.cuda()
            target = target.cuda()
            output = cnn(input)
            test_loss = loss(output, target)
            test_number += 1
            test_total_loss += test_loss.item()
            print("交叉验证集的Losss:{}".format(test_loss.item()))
            # accuracy = (output == labels.squeeze(1)).sum()
            # accuracy = (output.argmax(1) == labels).sum()
            # test_accuracy = test_accuracy + accuracy

    print("测试次数:{},测试数据集上的Loss:{}".format(epoch + 1, test_total_loss))
    # writer.add_scalar("test_number", test_number, epoch + 1)
    # writer.add_scalar("test_loss", test_total_loss, epoch + 1)
    writer.add_scalar("test_avg_loss", test_total_loss / test_number, epoch + 1)
    if epoch % 100 == 99:
        torch.save(cnn.state_dict(), "CNN_{}.pth".format(epoch + 1))
        print("模型已保存！")
writer.close()
    # scheduler.step()
# for epoch in range(epochs):
#     print("---------------------开始第{}轮网络训练-----------------".format(epoch + 1))
#     # 训练神经网络
#     for data in train_loader:
#         imgs, labelss = data
#         output = cnn(imgs)
#         target_loss = loss(output, labelss)
#         optim.zero_grad()
#         target_loss.backward()
#         optim.step()
#         train_step = train_step + 1
#         print("神经网络训练次数[{}], Loss:{}".format(train_step, target_loss.item()))
#         writer.add_scalar("train_loss", target_loss.item(), global_step=train_step)
#
#     # 测试神经网络
#     test_total_loss = 0
#     test_accuracy = 0
#     with torch.no_grad():
#         for data in test_loader:
#             imgs, labels = data
#             output = cnn(imgs)
#             test_loss = loss(output, labels)
#             test_total_loss = test_total_loss + test_loss
#             # accuracy = (output == labels.squeeze(1)).sum()
#             # accuracy = (output.argmax(1) == labels).sum()
#             # test_accuracy = test_accuracy + accuracy
#
#     print("测试次数:{},测试数据集上的Loss:{}".format(epoch+1, test_total_loss.item()))
#     # print("测试数据集上的acc:{}".format(test_accuracy / test_len))
#     # writer.add_scalar("test_accuracy", test_accuracy / test_len, test_step)
#     writer.add_scalar("test_loss", test_total_loss.item(), epoch)
#     # if epoch % 100 == 99:
#     torch.save(cnn.state_dict(), "CNN{}.pth".format(epoch+1))
#     print("模型已保存！")
#     # scheduler.step()


