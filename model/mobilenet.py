import torch
import torch.nn as nn

import numpy as np

import copy
from typing import List


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride: int = 1,
                ):
        super(DepthwiseSeparableConv, self).__init__()

        self.relu = nn.ReLU()

        self.depthwise_conv = nn.Conv2d(in_planes, in_planes, kernel_size=3, groups=in_planes, padding=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.pointwise_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False, stride=1)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pointwise_conv(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class MobileNetV1(nn.Module):
    def __init__(
        self,
        config: List[dict] = [
    {"in_planes": 32, "out_planes": 64, "stride":1, "repetition": 1},
    {"in_planes": 64, "out_planes": 128, "stride":2, "repetition": 1},
    {"in_planes": 128, "out_planes": 128, "stride":1, "repetition": 1},
    {"in_planes": 128, "out_planes": 256, "stride":2, "repetition": 1},
    {"in_planes": 256, "out_planes": 256, "stride":1, "repetition": 1},
    {"in_planes": 256, "out_planes": 512, "stride":2, "repetition": 1},
    {"in_planes": 512, "out_planes": 512, "stride":1, "repetition": 5},
    {"in_planes": 512, "out_planes": 1024, "stride":2, "repetition": 1},
    {"in_planes": 1024, "out_planes": 1024, "stride":2, "repetition": 1}
],
        width_multiplier: float = 1,
        resolution_multiplier: float = 1,
        num_classes: int = 10,
        block = DepthwiseSeparableConv
    ):
        super(MobileNetV1, self).__init__()
        if not config:
            raise Exception("Config is a required parameter")
        
        self.config = copy.deepcopy(config)
        self.width_multiplier = width_multiplier
        self.resolution_multiplier = resolution_multiplier
        self.num_classes= num_classes
        self.block = block
        
        for layer_config in self.config:
            layer_config["in_planes"] =  int(layer_config["in_planes"] * self.width_multiplier)
            layer_config["out_planes"] = int(layer_config["out_planes"] * self.width_multiplier)

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, self.config[0]["in_planes"], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.config[0]["in_planes"])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.config[-1]["out_planes"], self.num_classes)

        self.layer = self._make_layer(self.config)

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
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer(out)

        out = self.avg_pool(out)
        out = self.linear(out.view(out.shape[0], -1))
        
        return out

    def _make_layer(self, config):
        layer_list = []
        for layer_config in config:
            for _ in range(layer_config["repetition"]):
                layer = self.block(layer_config["in_planes"], layer_config["out_planes"], layer_config["stride"])
                layer_list.append(layer)

        return nn.Sequential(*layer_list)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        expansion_factor,
        input_channels,
        output_channels,
        stride
    ):
        super(InvertedResidual, self).__init__()


        bottleneck_width = int(input_channels * expansion_factor)
        
        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU6()

        self.conv1 = nn.Conv2d(input_channels, bottleneck_width, kernel_size=1, stride=1, bias=False)
        self.bn1 =  norm_layer(bottleneck_width)

        self.conv2 = nn.Conv2d(bottleneck_width, bottleneck_width, kernel_size=3, padding=1, stride=stride, bias=False, groups=bottleneck_width)
        self.bn2 =  norm_layer(bottleneck_width)

        self.conv3 = nn.Conv2d(bottleneck_width, output_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(output_channels)

        # Count model parameters
        self.total_params = 0
        for x in self.parameters():
            if x.requires_grad:
                self.total_params += np.prod(x.data.numpy().shape)
#        print(self.total_params)

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

        if identity.shape == out.shape:
            out += identity
        return out


class MobileNetV2(nn.Module):
    def __init__(
        self,
        Block =  InvertedResidual, 
        config = [
    {"expansion_factor": 1, "input_channels": 32, "output_channels": 16, "repeated": 1, "stride": 1},
    {"expansion_factor": 6, "input_channels": 16, "output_channels": 24, "repeated": 2, "stride": 2},
    {"expansion_factor": 6, "input_channels": 24, "output_channels": 32, "repeated": 3, "stride": 2},
    {"expansion_factor": 6, "input_channels": 32, "output_channels": 64, "repeated": 4, "stride": 2},
    {"expansion_factor": 6, "input_channels": 64, "output_channels": 96, "repeated": 3, "stride": 1},
    {"expansion_factor": 6, "input_channels": 96, "output_channels": 160, "repeated": 3, "stride": 2},
    {"expansion_factor": 6, "input_channels": 160, "output_channels": 320, "repeated": 1, "stride": 1},
],        
        num_classes: int = 1000
    ):
        super(MobileNetV2, self).__init__()

        self.config = copy.deepcopy(config)
        self.block = Block
        self.num_classes = num_classes

        self.relu = nn.ReLU6()
        self.norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, self.config[0]["input_channels"], kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = self.norm_layer(self.config[0]["input_channels"])

        self.layer = self._make_layer(self.config)

        last_conv_channel = int(self.config[-1]["output_channels"]*4)
        
        self.conv_last = nn.Conv2d(int(last_conv_channel/4), last_conv_channel, kernel_size=1, stride=1, bias=False)
        self.bn_last = self.norm_layer(last_conv_channel)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(last_conv_channel, self.num_classes, kernel_size=1, bias=False)        

        self.total_params = 0
        for x in self.parameters():
            if x.requires_grad:
                self.total_params += np.prod(x.data.numpy().shape)
        print(f"Number of model parameters: {self.total_params}")

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer(out)

        out = self.conv_last(out)
        out = self.bn_last(out)
        out = self.relu(out)

        out = self.avg_pool(out)
        out = self.classifier(out)
        out =  out.view((out.shape[0], out.shape[1]))
        return out

    def _make_layer(self, config):
        layer_list = []

        for layer_config in config:
            for idx in range(1, layer_config["repeated"] + 1):
                if idx == 1:
                    layer = self.block(
                        expansion_factor = layer_config["expansion_factor"],
                        input_channels = layer_config["input_channels"],
                        output_channels = layer_config["output_channels"],
                        stride = layer_config["stride"],
                    )
                    layer_list.append(layer)
                else:
                    layer = self.block(
                        expansion_factor = layer_config["expansion_factor"],
                        input_channels = layer_config["output_channels"],
                        output_channels = layer_config["output_channels"],
                        stride = 1,
                    )
                    layer_list.append(layer)

        return nn.Sequential(*layer_list)