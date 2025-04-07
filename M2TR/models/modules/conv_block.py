# 导入PyTorch深度学习相关模块
import torch.nn as nn  # 神经网络层定义
import torch.nn.functional as F  # 神经网络功能函数

'''
    定义了两个重要的神经网络模块
    1.Deconv（反卷积/上采样模块）
    想象这是一个"放大镜"，可以让图像特征变得更大、更清晰
    
    主要功能：
    将输入的特征图尺寸放大2倍
    使用双线性插值（类似于平滑缩放）
    通过卷积调整特征的通道数和细节
    使用LeakyReLU激活函数增加非线性


    2.ConvBN（卷积+批归一化模块）
    类似于一个"特征提取器"  
    
    主要功能：
    执行3x3卷积操作，提取图像特征
    使用批归一化（Batch Normalization）stabilize训练过程
    保持特征图大小不变
'''

class Deconv(nn.Module):
    """
    反卷积（上采样）模块
    
    主要功能：
    1. 对输入特征图进行上采样（放大）
    2. 通过卷积调整特征通道
    3. 使用LeakyReLU激活函数增加非线性
    
    参数：
    - input_channel: 输入特征图通道数
    - output_channel: 输出特征图通道数
    - kernel_size: 卷积核大小（默认3x3）
    - padding: 卷积填充大小（默认0）
    """
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        # 调用父类初始化方法
        super().__init__()
        
        # 创建卷积层
        self.conv = nn.Conv2d(
            input_channel,  # 输入通道数
            output_channel,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            stride=1,  # 步长为1
            padding=padding,  # 填充大小
        )

        # 创建LeakyReLU激活函数
        # 负半轴使用0.2的斜率，可以缓解梯度消失问题
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        前向传播方法
        
        处理流程：
        1. 使用双线性插值上采样（放大）输入特征图
        2. 通过卷积调整特征通道
        3. 使用LeakyReLU激活
        
        参数：
        - x: 输入特征图
        
        返回：
        - 处理后的特征图
        """
        # 使用双线性插值上采样，特征图尺寸扩大2倍
        x = F.interpolate(
            x, 
            scale_factor=2,  # 放大2倍
            mode='bilinear',  # 双线性插值
            align_corners=True  # 保持角点对齐
        )
        
        # 卷积处理
        out = self.conv(x)
        
        # 激活函数处理
        out = self.leaky_relu(out)
        
        return out


class ConvBN(nn.Module):
    """
    卷积+批归一化（Batch Normalization）模块
    
    主要功能：
    1. 执行卷积操作
    2. 对卷积输出进行批归一化
    
    参数：
    - in_features: 输入特征图通道数
    - out_features: 输出特征图通道数
    """
    def __init__(self, in_features, out_features):
        # 调用父类初始化方法
        super().__init__()
        
        # 创建卷积层
        # 3x3卷积核，padding=1保持特征图大小不变
        self.conv = nn.Conv2d(in_features, out_features, 3, padding=1)
        
        # 创建批归一化层
        # 对输出通道进行归一化，有助于稳定训练
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        """
        前向传播方法
        
        处理流程：
        1. 执行卷积操作
        2. 对卷积输出进行批归一化
        
        参数：
        - x: 输入特征图
        
        返回：
        - 处理后的特征图
        """
        # 卷积处理
        out = self.conv(x)
        
        # 批归一化
        out = self.bn(out)
        
        return out
