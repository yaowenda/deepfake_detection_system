# 导入必要的系统和深度学习库
import os  # 用于文件和路径操作

import torch  # PyTorch深度学习框架
from torch.utils.data import Dataset  # 基础数据集类


class DeepFakeDataset(Dataset):
    def __init__(
        self,
        dataset_cfg,  # 数据集配置字典
        mode='train',  # 数据集模式：训练、验证或测试
    ):
        # 1. 验证数据集名称
        dataset_name = dataset_cfg['NAME']  # 从配置中获取数据集名称
        assert dataset_name in [
            'ForgeryNet',  # 支持的数据集名称列表
            'FFDF',
            'CelebDF',
        ], 'no dataset'  # 如果数据集名称不在支持列表中，抛出异常

        # 2. 验证数据集模式
        assert mode in [
            'train',  # 支持的数据集模式
            'val',
            'test',
        ], 'wrong mode'  # 如果模式不在支持列表中，抛出异常

        # 3. 设置基本属性
        self.dataset_name = dataset_name  # 数据集名称
        self.mode = mode  # 数据集模式
        self.dataset_cfg = dataset_cfg  # 完整数据集配置
        self.root_dir = dataset_cfg['ROOT_DIR']  # 数据集根目录
        
        # 4. 确定信息文件路径
        info_txt_tag = mode.upper() + '_INFO_TXT'  # 生成配置中的信息文件标签（如TRAIN_INFO_TXT）

        # 5. 选择信息文件
        if dataset_cfg[info_txt_tag] != '':
            # 如果配置中直接指定了信息文件路径，使用该路径
            self.info_txt = dataset_cfg[info_txt_tag]
        else:
            # 否则，使用默认的路径生成规则
            self.info_txt = os.path.join(
                self.root_dir,
                self.dataset_name + '_splits_' + mode + '.txt',
            )
        
        # 6. 读取信息文件
        self.info_list = open(self.info_txt).readlines()  # 读取所有行

    # 数据集长度方法：返回样本总数
    def __len__(self):
        return len(self.info_list)

    # 标签转one-hot编码方法
    def label_to_one_hot(self, x, class_count):
        """
        将标签转换为one-hot编码
        
        参数:
        - x: 原始标签
        - class_count: 类别总数
        
        返回:
        - one-hot编码的标签张量
        """
        return torch.eye(class_count)[x.long(), :]  # 使用torch.eye生成one-hot编码
