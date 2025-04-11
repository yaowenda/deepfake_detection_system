import torch
from torch.utils.data import DataLoader


import M2TR.models 
import M2TR.datasets
import M2TR.utils.distributed as du
import M2TR.utils.logging as logging
from M2TR.utils.registries import (
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    MODEL_REGISTRY,
)

logger = logging.get_logger(__name__)


def build_model(cfg, gpu_id=None):
    # Construct the model
    model_cfg = cfg['MODEL']
    name = model_cfg['MODEL_NAME']
    logger.info('MODEL_NAME: ' + name)
    #MODEL_REGISTRY.get(name)会根据模型名称获取相应的构建器，然后 (model_cfg) 传递模型的配置信息给这个构建器，
    # 从而实例化模型对象并将其赋值给变量 model
    model = MODEL_REGISTRY.get(name)(model_cfg) 

    #判断是否可用
    assert torch.cuda.is_available(), "Cuda is not available."
    assert (
        cfg['NUM_GPUS'] <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"


    if gpu_id is None:
        # Determine the GPU used by the current process
        #获取当前进程默认使用的 CUDA 设备索引
        cur_device = torch.cuda.current_device()
    else:
        cur_device = gpu_id
    #将之前构建的模型 (model) 转移到指定的 CUDA 设备 (cur_device) 上
    model = model.cuda(device=cur_device)
    
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg['NUM_GPUS'] > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )

    return model


def build_loss_fun(cfg):
    loss_cfg = cfg['LOSS']
    name = loss_cfg['LOSS_FUN']
    logger.info('LOSS_FUN: ' + name)
    loss_fun = LOSS_REGISTRY.get(name)(loss_cfg)
    return loss_fun


def build_dataset(mode, cfg):
    dataset_cfg = cfg['DATASET']
    name = dataset_cfg['DATASET_NAME']
    logger.info('DATASET_NAME: ' + name + '  ' + mode)
    return DATASET_REGISTRY.get(name)(dataset_cfg, mode)


'''
    build_dataloader 函数的作用是根据给定的数据集、模式（训练或评估）和配置信息，
    创建一个适用于（特别是分布式）训练或评估的 PyTorch DataLoader。
'''
def build_dataloader(dataset, mode, cfg):
    dataloader_cfg = cfg['DATALOADER']
    num_tasks = du.get_world_size() #获取当前分布式训练环境中的总进程数
    global_rank = du.get_rank() #获取当前进程的全局排名


    '''
        DistributedSampler 是 PyTorch 提供的一个用于在分布式训练中对数据集进行采样的采样器。
        它的作用是确保每个训练进程只负责整个数据集的一个不重叠的子集，从而实现数据的并行加载和处理。
    '''
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True if mode == 'train' else False,
    )
    '''
        DataLoader 是 PyTorch 中用于加载数据的核心工具，它可以自动处理批处理、数据打乱、多进程加载等任务。
    '''
    return DataLoader(
        dataset,
        batch_size=dataloader_cfg['BATCH_SIZE'],
        sampler=sampler,
        num_workers=dataloader_cfg['NUM_WORKERS'], #数据加载的子进程数
        pin_memory=dataloader_cfg['PIN_MEM'],
        drop_last=True if mode == 'train' else False, #决定是否在最后一个批次的样本数量小于 batch_size 时将其丢弃
    )
