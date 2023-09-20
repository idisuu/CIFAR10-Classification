import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import numpy as np

import os
import json
import time
import argparse
from tqdm import tqdm

from model.resnet import ResNetCifar10, BasicBlock, Bottleneck, PreactivationBlock, ResNeXtBlock
from model.mobilenet import MobileNetV1, MobileNetV2

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def train(model):
    #dist.init_process_group("nccl")
    if use_ddp:
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")    
        device_id = rank % torch.cuda.device_count()
    else:
        device_id = "cuda:0"
        
    model.to(device_id)

    if use_ddp:
        ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False)  
    else:
        ddp_model = model
        rank = 0
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    loss_fct = nn.CrossEntropyLoss()
    
    iteration = 0

    train_loss_list = []
    train_acc_list = []
    
    test_loss_list = []
    test_acc_list = []    
    
    pbar = tqdm(total=max_iteration)
    while True:                
        ddp_model.train()
        temp_train_acc = []
        temp_train_loss = []
            
        start_time = time.time()
        for i in train_loader:
            
            input, label = i            
            input, label = input.to(device_id), label.to(device_id)
            out = ddp_model(input)
            loss = loss_fct(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            _, pred = torch.max(out, dim=-1)
            acc = (pred.cpu() == label.cpu()).numpy().mean()
    
            temp_train_loss.append(loss.item())
            temp_train_acc.append(acc)        
            
            iteration += 1
            pbar.update(1)
    
            if iteration in iteration_down_point:
                for param in optimizer.param_groups:
                    param['lr'] /= 10
                if rank == 0:
                    print()
                    print(f"LR ==> {optimizer.param_groups[0]['lr']}")
                    print()
                
            
            if iteration >= max_iteration:
                break
            
            temp_test_acc = []
            temp_test_loss =[]
                
            
            if iteration % validation_duration == 0:                
                ddp_model.eval()
                with torch.no_grad():
                    for i in test_loader:
                        input, label  = i
                        input, label = input.to(device_id), label.cuda(device_id)
                        out = ddp_model(input)                    
                        loss = loss_fct(out, label)
    
                        _, pred = torch.max(out, dim=-1)
                        acc = (pred.cpu() == label.cpu()).numpy().mean()
                                
                        temp_test_loss.append(loss.item())
                        temp_test_acc.append(acc)
                ddp_model.train()
    
                train_loss_list.append(np.array(temp_train_loss).mean())
                train_acc_list.append(np.array(temp_train_acc).mean())
                
                test_loss_list.append(np.array(temp_test_loss).mean())
                test_acc_list.append(np.array(temp_test_acc).mean())
                    
                end_time = time.time()               
                
                if (use_ddp and rank == 0) or (not use_ddp):
                    print()
                    print(f"{model_name} ==> Iteration {iteration}: train-loss - {train_loss_list[-1]}")
                    print(f"{model_name} ==> Iteration {iteration}: tess-acc - {test_acc_list[-1]}")
                    print(f'Take {round(end_time - start_time, 2)} for {validation_duration} iteration')
                    print()
                        
                start_time = time.time()
                    
        if iteration >= max_iteration:
            break
    if use_ddp:
        dist.destroy_process_group()    
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
    use_ddp = False
    if use_ddp:
        dist.init_process_group("nccl")
    world_size = 4
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    test_transfomrs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    
    root ="./data"
    train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=train_transforms, download=True)
    test_dataset =  torchvision.datasets.CIFAR10(root, train=False, transform=test_transfomrs, download=True)

    if use_ddp:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        train_shuffle= False
    else:
        train_sampler = None
        train_shuffle = True
    
    batch_size = 128
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=train_shuffle, sampler=train_sampler, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)
    
    num_classes = 10
    
    target_iteration = 64000
    max_iteration = target_iteration
    #max_iteration = int(target_iteration / world_size)
    
    validation_duration = 100
    iteration_down_point = [int(max_iteration*0.5), int(max_iteration*0.75)]
    
    save_folder = "./result/MobileNetV2/"    
    model_name = "MobileNetV2"
    save_mode = True

    if not os.path.exists(save_folder) and save_mode == True:
        os.makedirs(save_folder)
        print(f"{save_folder}가 생성되었습니다")

    if ("resnet" or "resnext")  in model_name.lower():
        #block_list = [BasicBlock, Bottleneck, PreactivationBlock, ResNeXtBlock]
        block = ResNeXtBlock
        num_layer = 3
        use_residual = True
        channel_list = [
            (88, 88),
            (88, 88),
            (88, 88)
        ]
        model  = ResNetCifar10(block, num_layer, num_classes, use_residual, channel_list).to(device_id)
        print(model.__class__.__name__)

    if ("mobilenetv1") in model_name.lower():
        model = MobileNetV1()
        print(model.__class__.__name__)

    if ("mobilenetv2") in model_name.lower():
        model = MobileNetV2()
        print(model.__class__.__name__)
    
    #model_name = block_to_model_name_dict[block.__name__] + f"_blocks_per_layer_{num_layer}"
    #model = ResNetCifar10(block, num_layer, num_classes, use_residual, channel_list)
    result = train(model)

    if save_mode:
        with open(f"{save_folder}{model_name}.json", 'w') as f:
            json.dump(result, f)        