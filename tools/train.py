import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import M2TR.utils.checkpoint as cu
import M2TR.utils.distributed as du
import M2TR.utils.logging as logging
from M2TR.utils.build_helper import (
    build_dataloader,
    build_dataset,
    build_loss_fun,
    build_model,
)
from M2TR.utils.meters import EpochTimer, MetricLogger, SmoothedValue
from M2TR.utils.optimizer import build_optimizer
from M2TR.utils.scheduler import build_scheduler
from tools.test import perform_test

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, criterion, optimizer, cfg, cur_epoch, cur_iter, writer
):
    '''
        - train_loader: 训练数据加载器
        - model: 深度学习模型
        - criterion: 损失函数
        - optimizer: 优化器
        - cfg: 配置参数
        - cur_epoch: 当前训练轮数
        - cur_iter: 当前迭代次数
        - writer: TensorBoard写入器
    '''
    # 训练准备
    model.train() # 设置模型为训练模式
    train_meter = MetricLogger(delimiter="  ") # 创建训练指标记录器
    train_meter.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}')) # 添加学习率记录
    header = 'Epoch: [{}]'.format(cur_epoch)
    print_freq = 10

    # 批次训练循环

    # 遍历train_loader中的每个批次的数据
    for samples in train_meter.log_every(train_loader, print_freq, header):
        #对每个批次的数据sample：提取字典中的所有张量，将这些张量移动到GPU，重新组合到一个新的字典，其中键保持不变，值已经位于 GPU 上。
        samples = dict(
            zip(
                samples,
                map(
                    lambda sample: sample.cuda(non_blocking=True),
                    samples.values(),
                ),
            )
        )
        # 前向传播
        with torch.cuda.amp.autocast(enabled=cfg['AMP_ENABLE']):
            outputs = model(samples) # 把samples输入到模型中，得到预测结果outputs
            loss = criterion(outputs, samples) # 预测结果和真实samples计算损失

        loss_value = loss.item() # 

        # 反向传播
        optimizer.zero_grad() # 把上一步的梯度清空
        loss.backward() # 计算当前损失对模型参数的梯度
        optimizer.step() # 用计算出的梯度更新模型参数（往更小的损失方向更新）

        torch.cuda.synchronize()

        # 梯度裁剪
        if 'CLIP_GRAD_L2NORM' in cfg['TRAIN']:  # TODO
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg['TRAIN']['CLIP_GRAD_L2NORM']
            )

        #记录日志与 TensorBoard 可视化
        if writer:
            writer.add_scalar('train loss', loss_value, global_step=cur_iter)
            writer.add_scalar(
                'lr', optimizer.param_groups[0]["lr"], global_step=cur_iter
            )
        train_meter.update(loss=loss_value)
        train_meter.update(lr=optimizer.param_groups[0]["lr"])
        cur_iter = cur_iter + 1
    # 更新迭代次数与指标
    train_meter.synchronize_between_processes()
    logger.info("Averaged stats:" + str(train_meter))

    # 返回每个指标的平均值与当前最新的迭代次数
    return {
        k: meter.global_avg for k, meter in train_meter.meters.items()
    }, cur_iter


def train(
    local_rank, num_proc, init_method, shard_id, num_shards, backend, cfg
):
    
    #分布式训练初始化
    world_size = num_proc * num_shards # 总进程数 = 每个节点进程数 × 节点数
    rank = shard_id * num_proc + local_rank  # 计算当前进程的全局排名
    torch.distributed.init_process_group( # 初始化分布式进程组
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank) # 设置当前进程使用的GPU

    # 环境设置
    du.init_distributed_training(cfg) # 初始化分布式训练环境
    np.random.seed(cfg['RNG_SEED']) # 设置随机种子确保可重复性
    torch.manual_seed(cfg['RNG_SEED']) # 设置PyTorch随机种子

    
    logging.setup_logging(cfg)
    logger.info(pprint.pformat(cfg))
    if du.is_master_proc(du.get_world_size()):
        writer = SummaryWriter(cfg['LOG_FILE_PATH'])
    else:
        writer = None

    #模型和数据准备
    model = build_model(cfg) # 构建模型
    optimizer = build_optimizer(model.parameters(), cfg) # 构建优化器
    scheduler, _ = build_scheduler(optimizer, cfg)  # TODO _?  # 构建学习率调度器
    loss_fun = build_loss_fun(cfg) # 构建损失函数
    train_dataset = build_dataset('train', cfg) # 构建训练数据集
    train_loader = build_dataloader(train_dataset, 'train', cfg) # 构建数据加载器
    val_dataset = build_dataset('val', cfg)
    val_loader = build_dataloader(val_dataset, 'val', cfg)

    start_epoch = cu.load_train_checkpoint(model, optimizer, scheduler, cfg)

    logger.info("Start epoch: {}".format(start_epoch + 1))
    epoch_timer = EpochTimer()

    cur_iter = 0

    #训练循环
    for cur_epoch in range(start_epoch, cfg['TRAIN']['MAX_EPOCH']): #从 start_epoch 开始（支持断点续训）到配置文件指定的最大轮数进行训练
        logger.info('========================================================')
        #为每个epoch设置不同的随机种子，确保分布式训练时数据的随机性和均匀性
        train_loader.sampler.set_epoch(cur_epoch)
        val_loader.sampler.set_epoch(cur_epoch)

        epoch_timer.epoch_tic() # 开始计时
        _, cur_iter = train_epoch( # 训练一个epoch
            train_loader,
            model,
            loss_fun,
            optimizer,
            cfg,
            cur_epoch,
            cur_iter,
            writer,
        )
        epoch_timer.epoch_toc() #结束计时

        perform_test(val_loader, model, cfg, cur_epoch, writer, mode='Val') # 在验证集上进行测试
        
        #性能统计输出
        #输出示例：Epoch 19 takes 1137.15s. Epochs from 0 to 19 take 1761.71s in average and 1229.09s in median.
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        #输出示例：For epoch 19, each iteraction takes 0.22s in average. From epoch 0 to 19, each iteraction takes 0.34s in average.
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        scheduler.step(cur_epoch) # 根据当前epoch更新学习率
        
        #判断是否需要保存检查点
        is_checkp_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch)

        # 如果是检查点epoch，保存模型状态
        if is_checkp_epoch:
            cu.save_checkpoint(model, optimizer, scheduler, cur_epoch, cfg)

    if writer: #训练结束后，确保TensorBoard写入器正确关闭
        writer.flush()
        writer.close()
        
        # 新增：

if __name__ == "__main__":
    import argparse
    import yaml
    from yacs.config import CfgNode
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=0, help="本地进程序号")
    args = parser.parse_args()
    
    # 加载默认配置
    cfg = CfgNode(new_allowed=True)
    default_cfg_path = "configs/default.yaml"
    cfg.merge_from_file(default_cfg_path)
    print(f"已加载默认配置: {default_cfg_path}")
    
    # 加载模型特定配置
    cfg.merge_from_file(args.cfg)
    print(f"已加载模型配置: {args.cfg}")
    
    # 打印最终配置
    print("最终配置:")
    print(cfg)
    
    # 调用训练函数
    train(
        local_rank=args.local_rank,
        num_proc=cfg.NUM_GPUS,
        init_method=cfg.INIT_METHOD,
        shard_id=cfg.SHARD_ID,
        num_shards=cfg.NUM_SHARDS,
        backend=cfg.DIST_BACKEND,
        cfg=cfg
    )