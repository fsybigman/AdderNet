#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
import torch
import adder
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
import math
import torch.nn as nn

def conv3x3(input_channel, output_channel, stride=1,op=1):
    " 3x3 convolution with padding op:1-add 0:mul"
    if op:
        return adder.adder2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False)
    else:
        return nn.Conv2d(input_channel, output_channel, kernel_size=3,stride=stride, padding=1, bias=False)

class Net(nn.Module):
    def __init__(self,op=1,num_classes=10):
        super(Net,self).__init__()
        self.conv1 = conv3x3(3,16,1,op)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv3x3(16,32,1,op)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = conv3x3(32,64,1,op)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64,num_classes,4,1,0)
        self.activate = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.activate(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.activate(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.conv4(x)
        return x.squeeze(-1).squeeze(-1)


parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--data', type=str, default='./data/')
parser.add_argument('--output_dir', type=str, default='./models/')
parser.add_argument('--load_path_n', type=int, default=-1)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  

acc = 0
acc_best = 0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_train = CIFAR10(args.data,
                   transform=transform_train,download = True)
data_test = CIFAR10(args.data,
                  train=False,
                  transform=transform_test)

data_train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=1)
data_test_loader = DataLoader(data_test, batch_size=16, num_workers=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

Addnet = Net(1,10).to(device)
Mulnet = Net(0,10).to(device)
if args.load_path_n != -1:
    Addnet = torch.load('./models/'+str(args.load_path_n)+'_addernet.pth', map_location=device)
    Mulnet = torch.load('./models/'+str(args.load_path_n)+'_mulnet.pth', map_location=device)

criterionadd = torch.nn.CrossEntropyLoss().to(device)
criterionmul = torch.nn.CrossEntropyLoss().to(device)

# optimizeradd = torch.optim.SGD(Addnet.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
# optimizermul = torch.optim.SGD(Mulnet.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizeradd = torch.optim.Adam(Addnet.parameters(), lr=0.0005)
optimizermul = torch.optim.Adam(Mulnet.parameters(), lr=0.0005)

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1+math.cos(float(epoch)/400*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    # adjust_learning_rate(optimizeradd, epoch)
    # adjust_learning_rate(optimizermul, epoch)

    for i, (images, labels) in enumerate(data_train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizeradd.zero_grad()
        optimizermul.zero_grad()
 
        outputadd = Addnet(images)
        outputmul = Mulnet(images)
 
        lossadd = criterionadd(outputadd, labels)
        lossmul = criterionmul(outputmul, labels)
 
        if i%50 == 0:
            print('ADD Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, lossadd.data.item()))
            print('MUL Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, lossmul.data.item()))
 
        lossadd.backward()
        optimizeradd.step()
        lossmul.backward()
        optimizermul.step()
 
 
# def test():
#     global acc, acc_best
#     net.eval()
#     total_correct = 0
#     avg_loss = 0.0
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(data_test_loader):
#             images, labels = Variable(images).to(device), Variable(labels).to(device)
#             output = net(images)
#             avg_loss += criterion(output, labels).sum()
#             pred = output.data.max(1)[1]
#             total_correct += pred.eq(labels.data.view_as(pred)).sum()
#
#     avg_loss /= len(data_test)
#     acc = float(total_correct) / len(data_test)
#     if acc_best < acc:
#         acc_best = acc
#     print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
 
 
def train_and_test(epoch):
    train(epoch)
    # test()

 
if __name__ == '__main__':
    epoch = 400
    for e in range(1, epoch):
        train_and_test(e)
        if epoch%5 == 0:
            torch.save(Addnet.state_dict(),args.output_dir + str(e) + '_addernet.pth')
            torch.save(Mulnet.state_dict(), args.output_dir + str(e) + '_mulnet.pth')
