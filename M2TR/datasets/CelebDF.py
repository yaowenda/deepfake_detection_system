import os

import torch

from M2TR.datasets.dataset import DeepFakeDataset
from M2TR.utils.registries import DATASET_REGISTRY

from .utils import get_image_from_path

'''
DATASET:
  NAME: CelebDF
  DATASET_NAME: CelebDF  # 数据集名称
  ROOT_DIR: /root/autodl-tmp/celeb-df-v2-img  # 数据集根目录
  TRAIN_INFO_TXT: '/root/autodl-tmp/test-splits-img/train.txt'  # 训练集信息文件
  VAL_INFO_TXT: '/root/autodl-tmp/test-splits-img/eval.txt'  # 验证集信息文件
  TEST_INFO_TXT: '/root/autodl-tmp/test-splits-img/test.txt'  # 测试集信息文件
  IMG_SIZE: 380  # 图像大小
  SCALE_RATE: 1.0  # 图像缩放比例
'''


@DATASET_REGISTRY.register()
class CelebDF(DeepFakeDataset):
    def __getitem__(self, idx):
        # 从信息列表中获取当前索引的行
        info_line = self.info_list[idx]
        # 去除换行符并分割信息
        image_info = info_line.strip('\n').split()
        # 获取图像路径
        image_path = image_info[1]
        # 构建完整路径
        image_abs_path = os.path.join(self.root_dir, image_path)

        img, _ = get_image_from_path(
            # 从路径加载图像并进行预处理
            image_abs_path, None, self.mode, self.dataset_cfg
        )
        img_label_binary = int(image_info[0])

        # 确保图像是正确的数据类型和范围
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        img = img.float()  # 转换为浮点类型

        # 构建样本字典：
        sample = {
            'img': img, # 处理后的图像
            'bin_label': [int(img_label_binary)], # 二分类标签
        }

        sample['img'] = torch.FloatTensor(sample['img']) # 将图像转换为浮点张量
        sample['bin_label'] = torch.FloatTensor(sample['bin_label']) # 将标签转换为浮点张量
        sample['bin_label_onehot'] = self.label_to_one_hot(
            sample['bin_label'], 2 # 2表示二分类
        ).squeeze() # 将标签转换为one-hot编码
        return sample