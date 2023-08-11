#!/usr/bin/env python3

import os
import argparse

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.models import vgg19

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader


def cifar_trainset(local_rank, dl_path='/tmp/cifar10-data'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def get_args():
    """命令行传参"""
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher 分布式环境下的当前进程号')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps 训练步数')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism  流水并行度')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend分布式后端')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)  # 添加deepspeed的配置参数
    args = parser.parse_args() # 解析参数
    return args


def train_base(args):
    torch.manual_seed(args.seed)

    # VGG also works :-)
    #net = vgg19(num_classes=10)
    net = AlexNet(num_classes=10) 

    trainset = cifar_trainset(args.local_rank) # 初始化训练集  若不存在，则自动下载

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    dataloader = RepeatingLoader(dataloader) # 转为无限循环的迭代器
    data_iter = iter(dataloader)

    rank = dist.get_rank() # 获取当前进程号
    gas = engine.gradient_accumulation_steps() # 获取梯度累积步数，模拟更大batch

    criterion = torch.nn.CrossEntropyLoss() # 损失函数

    total_steps = args.steps * engine.gradient_accumulation_steps() # 总步数=训练步数*梯度累积步数
    step = 0
    for micro_step in range(total_steps):
        batch = next(data_iter) # 获取一个batch的数据
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)

        outputs = engine(inputs)
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()

        if micro_step % engine.gradient_accumulation_steps() == 0:
            step += 1
            if rank == 0 and (step % 10 == 0):
                print(f'step: {step:3d} / {args.steps:3d} loss: {loss}')



def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1), # 补足漏掉的算子
        *vision_model.classifier,
    ]
    return layers


def train_pipe(args, part='parameters'):
    """流水并行"""
    # 设置随机种子，保证每次训练结果一致
    torch.manual_seed(args.seed) 
    deepspeed.runtime.utils.set_random_seed(args.seed)

    #
    # Build the model
    #

    # VGG also works :-)
    #net = vgg19(num_classes=10)
    net = AlexNet(num_classes=10) # 加载torchvision模型
    net = PipelineModule(layers=join_layers(net), # 模型序列
                         loss_fn=torch.nn.CrossEntropyLoss(), # 损失函数
                         num_stages=args.pipeline_parallel_size, # 流水并行度，即拆分为N段
                         partition_method=part, # 拆分方式，按照参数拆分或按照层拆分
                         activation_checkpoint_interval=0) # 梯度检查点，用于节省显存  0表示禁用

    trainset = cifar_trainset(args.local_rank)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    for step in range(args.steps):
        loss = engine.train_batch() # 流水并行必须使用engine.train_batch() 以减少气泡时间


if __name__ == '__main__':
    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend) # 初始化分布式环境
    args.local_rank = int(os.environ['LOCAL_RANK']) # 获取当前进程号
    torch.cuda.set_device(args.local_rank) # 设置当前进程使用的GPU

    if args.pipeline_parallel_size == 0:
        train_base(args) # DDP分布式数据并行
    else:
        train_pipe(args) # PP流水并行
