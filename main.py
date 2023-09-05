import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import numpy as np

import json
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, use_residual):
        super(BasicBlock, self).__init__()

        self.use_residual = use_residual
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU()

        self.shortcut = None
        if stride != 1 or (in_planes != planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            identity = self.shortcut(identity)

        if self.use_residual:
            out += identity
        out = self.relu(out)

        return out

class ResNetCifar10(nn.Module):
    def __init__(
        self,
        Block,
        num_layer: int,
        num_classes: int,
        use_residual: bool,
    ):
        super(ResNetCifar10, self).__init__()        
        self.block = Block
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.layer1 = self._make_layer(16, 16, num_layer, 1)
        self.layer2 = self._make_layer(16, 32, num_layer, 2)
        self.layer3 = self._make_layer(32, 64, num_layer, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = self.linear(out.view(out.shape[0], -1))
        return out

    def _make_layer(self, in_planes, planes, num_layer, stride):
        layers = []
        layers.append(
            self.block(in_planes, planes, stride, use_residual=self.use_residual)
        )

        for _ in range(1, num_layer):
            layers.append(
                self.block(
                    planes, planes, stride=1, use_residual=self.use_residual
                )
            )
        
        return nn.Sequential(*layers)

def train():
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    loss_fct = nn.CrossEntropyLoss()
    
    iteration = 0

    train_loss_list = []
    train_acc_list = []
    
    test_loss_list = []
    test_acc_list = []    
    
    while True:
        temp_train_acc = []
        temp_train_loss = []
        for i in train_loader:
            input, label = i
            input, label = input.cuda(), label.cuda()
            out = model(input)
            loss = loss_fct(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            _, pred = torch.max(out, dim=-1)
            acc = (pred.cpu() == label.cpu()).numpy().mean()
    
            temp_train_loss.append(loss.item())
            temp_train_acc.append(acc)        
            
            iteration += 1
    
            if iteration in iteration_down_point:
                for param in optimizer.param_groups:
                    param['lr'] /= 10
                print(f"LR ==> {optimizer.param_groups[0]['lr']}")
                
            
            if iteration >= max_iteration:
                break
            
            temp_test_acc = []
            temp_test_loss =[]
            if iteration % validation_duration == 0:
                with torch.no_grad():
                    for i in test_loader:
                        input, label  = i
                        input, label = input.cuda(), label.cuda()
                        out = model(input)                    
                        loss = loss_fct(out, label)
    
                        _, pred = torch.max(out, dim=-1)
                        acc = (pred.cpu() == label.cpu()).numpy().mean()
                                
                        temp_test_loss.append(loss.item())
                        temp_test_acc.append(acc)
    
                train_loss_list.append(np.array(temp_train_loss).mean())
                train_acc_list.append(np.array(temp_train_acc).mean())
                
                test_loss_list.append(np.array(temp_test_loss).mean())
                test_acc_list.append(np.array(temp_test_acc).mean())

                print(f"{model_name} ==> Iteration {iteration}: {train_loss_list[-1]}")
                    
        if iteration >= max_iteration:
            break
    return {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'test_loss': test_loss_list,
        'test_acc': test_acc_list
    }

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transfomrs = transforms.Compose([
    transforms.ToTensor()
])

root ="./data"
train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=train_transforms, download=True)
test_dataset =  torchvision.datasets.CIFAR10(root, train=False, transform=test_transfomrs, download=True)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

Block = BasicBlock
num_classes = 10

max_iteration = 64000
validation_duration = 100
iteration_down_point = [32000, 48000]

repetition = [3, 9]
use_residual_list = [True, False]
for num_layer in tqdm(repetition):
    for use_residual in use_residual_list:
        model_name = f"ResNet-{num_layer*6+2}-{use_residual}"
        
        model = ResNetCifar10(Block, num_layer, num_classes, use_residual)
        model.cuda()
        result = train()
        with open(f"./result/{model_name}.json", 'w') as f:
            json.dump(result, f)        