# 导入必要的系统和图像处理库
import os  # 文件和路径操作
import random  # 随机数生成

# 图像增强和深度学习相关库
import albumentations  # 先进的图像增强库
import albumentations.pytorch  # PyTorch集成
import numpy as np  # 数值计算
import torchvision  # 计算机视觉工具库
from PIL import Image  # 图像处理

# 图像随机裁剪和缩放类：增强数据多样性
class ResizeRandomCrop:
    """
    图像随机裁剪和缩放处理类
    
    主要功能：
    1. 按概率对图像进行随机缩放和裁剪
    2. 保持图像大小一致
    3. 可选是否处理掩码图像
    
    参数：
    - img_size: 目标图像大小（默认320）
    - scale_rate: 缩放比率（默认8/7）
    - p: 随机裁剪的概率（默认0.5）
    """
    def __init__(self, img_size=380, scale_rate=8 / 7, p=0.5):
        self.img_size = img_size  # 目标图像大小
        self.scale_rate = scale_rate  # 缩放比率
        self.p = p  # 随机裁剪概率

    def __call__(self, image, mask=None):
        """
        实际的图像处理方法
        
        处理流程：
        1. 根据概率决定是否进行随机裁剪
        2. 如果进行裁剪：
           - 先按比例放大图像
           - 再随机裁剪到目标大小
        3. 如果不进行裁剪：
           - 直接缩放到目标大小
        
        参数：
        - image: 输入图像
        - mask: 可选的掩码图像
        
        返回：
        - 处理后的图像和掩码
        """
        # 随机决定是否进行裁剪
        if random.uniform(0, 1) < self.p:
            # 按比例计算放大尺寸
            S1 = int(self.img_size * self.scale_rate)
            S2 = S1
            
            # 创建缩放函数
            resize_func = torchvision.transforms.Resize((S1, S2))
            
            # 放大图像
            image = resize_func(image)
            
            # 获取随机裁剪参数
            crop_params = torchvision.transforms.RandomCrop.get_params(
                image, (self.img_size, self.img_size)
            )
            
            # 裁剪图像
            image = torchvision.transforms.functional.crop(image, *crop_params)
            
            # 如果有掩码，也进行相同的处理
            if mask is not None:
                mask = resize_func(mask)
                mask = torchvision.transforms.functional.crop(
                    mask, *crop_params
                )

        else:
            # 如果不进行随机裁剪，直接缩放到目标大小
            resize_func = torchvision.transforms.Resize(
                (self.img_size, self.img_size)
            )
            image = resize_func(image)
            
            # 掩码也同样处理
            if mask is not None:
                mask = resize_func(mask)

        return image, mask


def transforms_mask(mask_size):
    """
    创建掩码图像的转换函数
    
    参数：
    - mask_size: 掩码图像的目标大小
    
    返回：
    - 掩码图像转换管道
    """
    return albumentations.Compose(
        [
            albumentations.Resize(mask_size, mask_size),  # 调整大小
            albumentations.pytorch.transforms.ToTensorV2(),  # 转换为PyTorch张量
        ]
    )


def get_augmentations_from_list(augs: list, aug_cfg, one_of_p=1):
    """
    从配置列表生成图像增强操作
    
    参数：
    - augs: 增强操作列表
    - aug_cfg: 增强配置
    - one_of_p: 随机选择概率
    
    返回：
    - 处理后的增强操作列表
    """
    ops = []
    for aug in augs:
        if isinstance(aug, list):
            # 如果是嵌套列表，使用OneOf随机选择
            op = albumentations.OneOf
            param = get_augmentations_from_list(aug, aug_cfg)
            param = [param, one_of_p]
        else:
            # 获取具体的增强操作
            op = getattr(albumentations, aug)
            
           # 获取操作参数
            if aug.upper() + '_PARAMS' in aug_cfg:
                param = aug_cfg[aug.upper() + '_PARAMS']
                # 如果是需要height和width的操作，从IMG_SIZE获取
                if 'Resize' in aug or 'Crop' in aug:
                    if 'height' not in param and 'width' not in param:
                        img_size = aug_cfg.get('IMG_SIZE', 320)
                        param['height'] = img_size
                        param['width'] = img_size
            else:
                param = {}
        
        # 添加操作
        ops.append(op(*tuple(param)))
    return ops


def get_transformations(mode, dataset_cfg):
    """根据模式获取图像转换管道"""
    # 选择数据增强配置
    aug_list = dataset_cfg['AUGMENTATIONS']['TRAIN' if mode == 'train' else 'TEST']
    
    ops = []
    for aug in aug_list:
        if isinstance(aug, dict):
            # 获取增强操作名称和参数
            aug_name = list(aug.keys())[0]
            aug_params = list(aug.values())[0]
            # 创建增强操作
            op = getattr(albumentations, aug_name)
            ops.append(op(**aug_params))
    
    # 添加转换为张量的操作
    ops.append(albumentations.pytorch.transforms.ToTensorV2())
    
    # 创建完整的增强管道
    return albumentations.Compose(ops, p=1)


def get_image_from_path(img_path, mask_path, mode, dataset_cfg):
    """
    从路径加载图像并进行预处理
    
    参数：
    - img_path: 图像路径
    - mask_path: 掩码图像路径
    - mode: 数据集模式
    - dataset_cfg: 数据集配置
    
    返回：
    - 处理后的图像和掩码
    """
    # 获取图像配置
    img_size = dataset_cfg['IMG_SIZE']
    scale_rate = dataset_cfg['SCALE_RATE']

    # 加载图像
    img = Image.open(img_path)
    
    # 加载掩码（如果存在）
    if mask_path is not None and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')
    else:
        # 如果没有掩码，创建全零掩码
        mask = Image.fromarray(np.zeros((img_size, img_size)))

    # 获取转换管道
    trans_list = get_transformations(
        mode,
        dataset_cfg,
    )

    # 训练模式下的处理
    if mode == 'train':
        # 随机裁剪
        crop = ResizeRandomCrop(img_size=img_size, scale_rate=scale_rate)
        img, mask = crop(image=img, mask=mask)

        # 转换图像
        img = np.asarray(img)
        img = trans_list(image=img)['image']

        # 转换掩码
        mask = np.asarray(mask)
        mask = transforms_mask(img_size)(image=mask)['image']

    else:
        # 测试模式下的处理
        img = np.asarray(img)
        img = trans_list(image=img)['image']
        mask = np.asarray(mask)
        mask = transforms_mask(img_size)(image=mask)['image']
        
    # 最后调整到320x320（如果需要）
    if img_size != 320:
        resize_to_model = torchvision.transforms.Resize((320, 320))
        img = resize_to_model(img)
        if mask is not None:
            mask = resize_to_model(mask)

    return img, mask.float()


def get_mask_path_from_img_path(dataset_name, root_dir, img_info):
    """
    根据图像路径获取对应的掩码路径
    
    参数：
    - dataset_name: 数据集名称
    - root_dir: 数据集根目录
    - img_info: 图像信息
    
    返回：
    - 掩码图像的完整路径
    """
    # ForgeryNet数据集的掩码路径处理
    if dataset_name == 'ForgeryNet':
        root_dir = os.path.join(root_dir, 'spatial_localize')
        fore_path = img_info.split('/')[0]
        
        # 根据训练/测试调整路径
        if 'train' in fore_path:
            img_info = img_info.replace('train_release', 'train_mask_release')
        else:
            img_info = img_info[20:]

        mask_complete_path = os.path.join(root_dir, img_info)

    # FFDF数据集的掩码路径处理
    elif 'FFDF' in dataset_name:
        mask_info = img_info.replace('images', 'masks')
        mask_complete_path = os.path.join(root_dir, mask_info)

    return mask_complete_path
