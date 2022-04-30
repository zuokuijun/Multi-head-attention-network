import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from Data import *
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            # 第二层卷积
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            # 第三层卷积
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 第四层卷积
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 16),
            nn.Tanh(),

        )
        self.layer2 = nn.Sequential(
            nn.Linear(16, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 70)
        )

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        return output2

if __name__ == '__main__':
    model = CNN()
    print(model)
    # img_dir = "../data/naca/images/"
    # label_dir = "../data/naca/labels/"
    # data = MyData(img_dir, label_dir)
    # imges, labels = data[0]
    # imges =torch.reshape(imges,(1,4,216,216))
    model.load_state_dict(torch.load("./CNN4900.pth", map_location=torch.device('cpu')))
    input = torch.ones(1, 1, 216, 216)
    out = model(input)
    print(out)
    # print(imges)
    # print(labels.shape)