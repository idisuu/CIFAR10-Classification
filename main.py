import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import numpy as np

import os
import json
from tqdm.notebook import tqdm

from model.resnet import ResNetCifar10, BasicBlock, Bottleneck, PreactivationBlock, ResNeXtBlock


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


block_to_model_name_dict = {
    'BasicBlock': "ResNet",
    "Bottleneck": "ResNet-Bottleneck",
    "PreactivationBlock": "ResNet-V2",
    "ResNeXtBlock": "ResNext"
}

if __name__ == "__main__":    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transfomrs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    root ="./data"
    train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=train_transforms, download=True)
    test_dataset =  torchvision.datasets.CIFAR10(root, train=False, transform=test_transfomrs, download=True)
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    num_classes = 10
    
    max_iteration = 64000
    validation_duration = 100
    iteration_down_point = [32000, 48000]
    channel_list = [
        (64, 64),
        (64, 64),
        (64, 64)
    ]
    save_folder = "./result/block_evaluation/"
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"{save_folder}가 생성되었습니다")

    #block_list = [BasicBlock, Bottleneck, PreactivationBlock, ResNeXtBlock]
    block_list = [BasicBlock, PreactivationBlock]
    repetition = [18]
    use_residual_list = [True]
    
    for block in block_list:
        for num_layer in repetition:
            for use_residual in use_residual_list:
                model_name = block_to_model_name_dict[block.__name__] + f"_blocks_per_layer_{num_layer}"
                
                model = ResNetCifar10(block, num_layer, num_classes, use_residual, channel_list)
                model.cuda()
                result = train()
                with open(f"{save_folder}{model_name}.json", 'w') as f:
                    json.dump(result, f)        