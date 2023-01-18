import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

def Conv(in_channels, out_channels, kerner_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kerner_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


class TinyAge(nn.Module):
    def __init__(self):
        super(TinyAge, self).__init__()
        self.conv1 = Conv(3, 16, 3, 1, 1)
        self.conv2 = Conv(16, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv(16, 32, 3, 1, 1)
        self.conv4 = Conv(32, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv(32, 64, 3, 1, 1)
        self.conv6 = Conv(64, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = Conv(64, 128, 3, 1, 1)
        self.conv8 = Conv(128, 128, 3, 1, 1)
        self.conv9 = Conv(128, 128, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv10 = Conv(128, 128, 3, 1, 1)
        self.conv11 = Conv(128, 128, 3, 1, 1)
        self.conv12 = Conv(128, 128, 3, 1, 1)
        self.HP = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128, 101),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.pool5(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.HP(x)
        x = x.view((x.size(0), -1))
        x = self.fc1(x.view((x.size(0), -1)))
        x = F.normalize(x, p=1, dim=1)

        return x

class ThinAge(nn.Module):
    def __init__(self):
        super(ThinAge, self).__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)
        self.conv2 = Conv(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv(32, 64, 3, 1, 1)
        self.conv4 = Conv(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv(64, 128, 3, 1, 1)
        self.conv6 = Conv(128, 128, 3, 1, 1)
        self.conv7 = Conv(128, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = Conv(128, 256, 3, 1, 1)
        self.conv9 = Conv(256, 256, 3, 1, 1)
        self.conv10 = Conv(256, 256, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv11 = Conv(256, 256, 3, 1, 1)
        self.conv12 = Conv(256, 256, 3, 1, 1)
        self.conv13 = Conv(256, 256, 3, 1, 1)
        self.HP = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 101),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.HP(x)
        x = x.view((x.size(0), -1))
        x = self.fc(x.view((x.size(0), -1)))
        x = F.normalize(x, p=1, dim=1)

        return x


#使用官方的模型预测
def preprocess(img):
    img = Image.open(img).convert('RGB')
    imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
    for x in imgs:
        plt.imshow(x)
        plt.show()

    transform_list = [
        transforms.Resize((224,224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    imgs = [transform(i) for i in imgs]
        # plt.show()
    imgs = [torch.unsqueeze(i, dim=0) for i in imgs]
    return imgs



def test(dir,paths):
    model = ThinAge()
    model.load_state_dict(torch.load('pretrained/ThinAge_dict.pt'))  # 这里根据模型结构，调用存储的模型参数

    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    rank = torch.Tensor([i for i in range(101)]).cuda()

    for img_path in paths:
        imgs = preprocess(img_path)
        predict_age = 0
        #for img in imgs:
        img = imgs[0].to(device)

        output = model(img)
        print(output)
        predict_age= torch.sum(output*rank, dim=1).item()/2
        print(img_path,predict_age)



# if __name__ == '__main__':
dirs = os.listdir('NPR_test')
dirs = ['./NPR_test/' + x for x in dirs]
print(dirs)
for dir in dirs:
    pics=os.listdir(dir)
    pics=[dir+'/'+x for x in pics]
    test(dir,pics)
    break
