# 导入PyTorch深度学习相关模块
import torch.nn as nn


class GramMatrix(nn.Module):
    """
    Gram矩阵计算模块 - 风格特征提取器
    
    Gram矩阵是一种特殊的矩阵，用于捕捉图像特征的统计信息和纹理特征

    类似于分析图像的"纹理指纹"
    
    主要功能：
    1. 将输入特征图转换为Gram矩阵
    2. 捕捉特征通道之间的相关性
    3. 在风格迁移和图像生成任务中广泛使用
    
    工作原理：
    - 将特征图重塑为二维矩阵
    - 计算特征矩阵与其转置的矩阵乘积
    - 得到反映特征通道间关系的Gram矩阵
    """
    def __init__(self):
        # 调用父类初始化方法
        super(GramMatrix, self).__init__()

    def forward(self, x):
        """
        前向传播方法
        
        参数：
        - x: 输入特征图，形状为 (batch_size, channels, height, width)
        
        返回：
        - Gram矩阵，形状为 (batch_size, 1, height, width)
        """
        # 获取输入特征图的维度
        b, c, h, w = x.size()
        
        # 将特征图重塑为 (batch_size, channels, height*width)
        feature = x.view(b, c, h * w)
        
        # 转置特征矩阵，用于后续计算
        feature_t = feature.transpose(1, 2)
        
        # 计算Gram矩阵：特征矩阵与其转置的矩阵乘积
        # 这一步捕捉特征通道间的相关性
        gram = feature.bmm(feature_t)
        
        # 获取Gram矩阵的维度
        b, h, w = gram.size()
        
        # 重塑Gram矩阵为 (batch_size, 1, height, width)
        # 便于后续卷积和处理
        gram = gram.view(b, 1, h, w)
        
        return gram


class GramBlock(nn.Module):
    """
    Gram块 - 复杂的特征提取和池化模块
    
    主要功能：
    1. 通过一系列卷积层提取和转换图像特征
    2. 使用Gram矩阵捕捉特征的统计信息
    3. 通过多层卷积和池化压缩特征
    
    适用场景：
    - 深度伪造检测
    - 图像风格分析
    - 纹理特征提取
    """
    def __init__(self, in_channels):
        """
        初始化方法
        
        参数：
        - in_channels: 输入特征图的通道数
        """
        # 调用父类初始化方法
        super(GramBlock, self).__init__()
        
        # 第一层卷积：初步提取特征
        # 将输入通道转换为32个通道
        # 使用较大的padding保持特征图尺寸
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=2
        )
        
        # 创建Gram矩阵模块
        self.gramMatrix = GramMatrix()
        
        # 第二层卷积：处理Gram矩阵
        # 包含卷积、批归一化和ReLU激活
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=2),  # 减半特征图尺寸
            nn.BatchNorm2d(16),  # 标准化特征
            nn.ReLU(inplace=True),  # 非线性激活
        )
        
        # 第三层卷积：进一步压缩和提取特征
        # 继续减半特征图尺寸
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 自适应平均池化：将特征压缩为1x1
        # 无论输入大小，都输出固定大小的特征向量
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        前向传播方法
        
        处理流程：
        1. 初步卷积提取特征
        2. 计算Gram矩阵
        3. 多层卷积和池化压缩特征
        
        参数：
        - x: 输入特征图
        
        返回：
        - 压缩后的特征表示
        """
        # 第一层卷积
        x = self.conv1(x)
        
        # 计算Gram矩阵
        x = self.gramMatrix(x)
        
        # 第二层卷积
        x = self.conv2(x)
        
        # 第三层卷积
        x = self.conv3(x)
        
        # 自适应平均池化，压缩为1x1特征
        x = self.pool(x)
        
        return x
