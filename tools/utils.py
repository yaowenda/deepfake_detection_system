import argparse

import torch
import yaml

import M2TR.utils.logging as logging
from M2TR.utils.checkpoint import get_path_to_checkpoint

logger = logging.get_logger(__name__)


def parse_args(): # 这个函数的作用是解析命令行的参数，并返回解析后的结果
    parser = argparse.ArgumentParser( # 创建一个解析器对象 parser
        #参数 description 提供了一个简短的描述，说明这个脚本的功能是“提供训练和测试的管道”。
        description="Provide training and testing pipeline."
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_file", # 将解析后的值存储到 args 对象的 cfg_file 属性中（即 args.cfg_file）
        help="Path to the config file", #提供帮助信息，说明该参数的作用是“配置文件的路径”
        required=True,
        type=str,
    )
    parser.add_argument(
        "--shard_id",
        dest="shard_id", #将解析后的值存储到 args 对象的 shard_id 属性中（即 args.shard_id）
        #提供帮助信息，说明该参数表示当前节点的分片 ID，范围是从 0 到 NUM_SHARDS - 1
        help="The shard id of current node, Starts from 0 to NUM_SHARDS - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--lr",
        dest="base_lr", #将解析后的值存储到 args 对象的 base_lr 属性中（即 args.base_lr）
        help="The base learning rate", #提供帮助信息，说明该参数表示基础学习率
        type=float,
    )
    #调用 parser.parse_args() 方法解析命令行参数，返回一个包含所有解析后参数的对象 args，用户可以通过 args.cfg_file等访问这些参数的值
    return parser.parse_args() 

# 合并两个字典（合并两个配置文件） 自定义配置会覆盖默认配置中相同的键值。
def merge_a_into_b(a, b):
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def load_config(args):
    with open('./configs/default.yaml', 'r') as file: #这里加载的是default.yaml这个配置文件
        cfg = yaml.safe_load(file)
    logger.info('Use cfg_file: ', './configs/' + args.cfg_file)
    with open('./configs/' + args.cfg_file, 'r') as file:  # 这里加载的是m2tr.yaml这个配置文件 因为命令行输入的参数是--cfg m2tr.yaml
        custom_cfg = yaml.safe_load(file)
    merge_a_into_b(custom_cfg, cfg)
    if args.shard_id is not None:
        cfg['SHARD_ID'] = args.shard_id
    if args.base_lr is not None:
        cfg['OPTIMIZER']['BASE_LR'] = args.base_lr #OPTIMIZER（优化器）的BASE_LR（基础学习率）默认设置为0.001，可自行设置

    if cfg['TRAIN']['ENABLE']:
        cfg['TEST']['CHECKPOINT_TEST_PATH'] = get_path_to_checkpoint(
            cfg['TRAIN']['CHECKPOINT_SAVE_PATH'], cfg['TRAIN']['MAX_EPOCH'], cfg
        )
    cfg['DATASET']['TRAIN_AUGMENTATIONS']['RESIZE_PARAMS'] = [
        cfg['DATASET']['IMG_SIZE'], #380
        cfg['DATASET']['IMG_SIZE'],
    ]
    cfg['DATASET']['TEST_AUGMENTATIONS']['RESIZE_PARAMS'] = [
        cfg['DATASET']['IMG_SIZE'],
        cfg['DATASET']['IMG_SIZE'],
    ]

    logger.info(cfg)
    return cfg


def launch_func(cfg, func, daemon=False): # cfg是配置字典 func是要执行的训练函数，在这里有两个选择 train和test
    if cfg['NUM_GPUS'] > 1: # 多GPU时
        torch.multiprocessing.spawn( # 启动多个进程
            func,
            nprocs=cfg['NUM_GPUS'], # nprocs指定进程数量 是GPU数量
            args=( #传递给每个进程的参数
                cfg['NUM_GPUS'], # GPU总数
                cfg['INIT_METHOD'], # 进程初始化方法
                cfg['SHARD_ID'], # 当前节点ID
                cfg['NUM_SHARDS'], # 总节点数
                cfg['DIST_BACKEND'], # 分布式后端（如 nccl, gloo 等）
                cfg,
            ),
            daemon=daemon,
        )
    else: # 单GPU时
        func(
            local_rank=0, # 本地进程号
            num_proc=1, # 进程数量
            init_method=cfg['INIT_METHOD'], # INIT_METHOD: 'tcp://localhost:9999'
            shard_id=0, # 当前节点ID
            num_shards=1, # 总节点数
            backend=cfg['DIST_BACKEND'], # 分布式后端
            cfg=cfg,
        )

def get_image_from_path(img_path, mask_path, mode, dataset_cfg, is_image=False):
    # 如果是图像对象
    if is_image:
        img = img_path  # img_path 实际上是 PIL Image 对象
    else:
        img = Image.open(img_path)