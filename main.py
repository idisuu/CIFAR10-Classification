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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def train():
    #dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    
    device_id = rank % torch.cuda.device_count()
    model  = ResNetCifar10(block, num_layer, num_classes, use_residual, channel_list).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False)    
    
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
                
                if rank == 0:
                    print()
                    print(f"{model_name} ==> Iteration {iteration}: train-loss - {train_loss_list[-1]}")
                    print(f"{model_name} ==> Iteration {iteration}: tess-acc - {test_acc_list[-1]}")
                    print(f'Take {round(end_time - start_time, 2)} for {validation_duration} iteration')
                    print()
                        
                start_time = time.time()
                    
        if iteration >= max_iteration:
            break
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
    dist.init_process_group("nccl")
    world_size = 4
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transfomrs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    root ="./data"
    train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=train_transforms, download=True)
    test_dataset =  torchvision.datasets.CIFAR10(root, train=False, transform=test_transfomrs, download=True)
    
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    
    batch_size = 256
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=train_sampler, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)
    
    num_classes = 10
    
    target_iteration = 64000
    max_iteration = target_iteration
    #max_iteration = int(target_iteration / world_size)
    
    validation_duration = 100
    iteration_down_point = [int(max_iteration*0.5), int(max_iteration*0.75)]
    channel_list = [
        (16, 16),
        (16, 32),
        (32, 64)
    ]
    save_folder = "./result/iteration_test/"
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"{save_folder}가 생성되었습니다")

    #block_list = [BasicBlock, Bottleneck, PreactivationBlock, ResNeXtBlock]
    block = BasicBlock
    num_layer = 3
    use_residual = True
    
    #model_name = block_to_model_name_dict[block.__name__] + f"_blocks_per_layer_{num_layer}"
    model_name = f"iteration-{max_iteration}_batchsize-{batch_size}_worldsize-{world_size}_sampler-true"
    #model = ResNetCifar10(block, num_layer, num_classes, use_residual, channel_list)
    result = train()
                
    with open(f"{save_folder}{model_name}.json", 'w') as f:
        json.dump(result, f)        