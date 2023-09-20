import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import List

class BasicBlock(nn.Module):
    expansion = 1
    preactivation = False
    def __init__(self, in_planes, planes, stride, use_residual):
        super(BasicBlock, self).__init__()

        self.use_residual = use_residual
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
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


class Bottleneck(nn.Module):
    expansion = 4
    preactivation = False
    def __init__(self, in_planes, planes, stride, use_residual=True):
        super(Bottleneck, self).__init__()
        self.use_residual = use_residual
        
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_planes*self.expansion, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.shortcut = None
        if (in_planes != planes) or (stride != 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes*self.expansion, planes*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):
        print_mode = False
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut:
            identity = self.shortcut(identity)

        if self.use_residual:
            out += identity
            
        out = self.relu(out)
        return out


class PreactivationBlock(nn.Module):
    expansion = 1
    preactivation = True
    def __init__(self, in_planes, planes, stride, use_residual):
        super(PreactivationBlock, self).__init__()        
        self.use_residual = use_residual

        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 =  nn.BatchNorm2d(in_planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if (in_planes != planes) or (stride != 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )        

    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)       

        if self.shortcut:
            identity = self.shortcut(identity)

        if self.use_residual:
            out += identity               
        return out


class ResNeXtBlock(nn.Module):
    expansion = 4
    preactivation = False
    def __init__(self, in_planes, planes, stride, cardinality=2, use_residual=True):
        super(ResNeXtBlock, self).__init__()

        #self.planes = int(planes * self.expansion / 2)
        self.use_residual = use_residual
        
        self.relu = nn.ReLU()

        self.bottleneck_width = int(planes*cardinality)
        
        self.conv1 = nn.Conv2d(in_planes*self.expansion, self.bottleneck_width, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(self.bottleneck_width)
        
        self.conv2 =  nn.Conv2d(self.bottleneck_width, self.bottleneck_width, kernel_size=3, stride=1, bias=False, padding=1, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_width)
        
        self.conv3 = nn.Conv2d(self.bottleneck_width, planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.shortcut = None
        if (in_planes*self.expansion != planes*self.expansion) or (stride != 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes*self.expansion, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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
        channel_list: List[tuple] = [(16, 16),
                                     (16, 32),
                                     (32, 64)]
    ):
        super(ResNetCifar10, self).__init__()
        
        self.block = Block
        for (in_planes, planes) in channel_list:
            if (in_planes % self.block.expansion) or (planes % self.block.expansion):
                raise ValueError(f"in_planes({in_planes}) and planes({planes}) should be divided by block.expansion({self.block.expansion})")
 
        self.use_residual = use_residual
        self.channel_list = channel_list
        self.layer_order = 1

        self.conv1 = nn.Conv2d(3, channel_list[0][0], kernel_size=3, padding=1)
        if not self.block.preactivation:
            self.bn1 = nn.BatchNorm2d(channel_list[0][0])
        
        self.layer1 = self._make_layer(channel_list[0][0], channel_list[0][1], num_layer, 1)
        self.layer2 = self._make_layer(channel_list[1][0], channel_list[1][1], num_layer, 2)
        self.layer3 = self._make_layer(channel_list[2][0], channel_list[2][1], num_layer, 2)
        
        if self.block.preactivation:
            self.bn_last = nn.BatchNorm2d(64*self.block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(channel_list[-1][1]*self.block.expansion, num_classes)

        self.relu = nn.ReLU()
        
        # weight 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Calculate number of parameters
        self.total_params = 0    
        for x in filter(lambda p: p.requires_grad, self.parameters()):
            self.total_params += np.prod(x.data.numpy().shape)

        print(f'Total parameters: {self.total_params}')

    def forward(self, x):
        out = self.conv1(x)
        if not self.block.preactivation:
            out = self.bn1(out)
            out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.block.preactivation:
            out = self.bn_last(out)
            out = self.relu(out)
        
        out = self.avgpool(out)
        out = self.linear(out.view(out.shape[0], -1))
        return out

    def _make_layer(self, in_planes, planes, num_layer, stride):
        if (self.block.expansion > 1) and (self.channel_list[0][0] == in_planes) and (self.channel_list[0][1] == planes) and (self.block.__name__ != 'ResNeXtBlock'):
            in_planes = int(in_planes/self.block.expansion)

        # Custom Configuration for ResNeXt CIFAR10
        if self.block.__name__ == 'ResNeXtBlock' and self.layer_order == 1:
            in_planes = int(self.channel_list[0][0] / self.block.expansion)
            
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
        self.layer_order += 1        
        
        return nn.Sequential(*layers)