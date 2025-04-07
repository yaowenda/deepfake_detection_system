# 导入PyTorch深度学习相关模块
import torch.nn as nn


class Mlp(nn.Module):
    """
    多层感知器（Multi-Layer Perceptron）模块
    
    主要功能：
    1. 对输入特征进行非线性变换
    2. 通过两个全连接层和激活函数增加模型复杂度
    3. 使用Dropout防止过拟合
    
    适用场景：
    - Transformer模型中的前馈神经网络
    - 特征空间的非线性映射
    - 增加模型表达能力
    """
    def __init__(
        self,
        in_features,  # 输入特征维度
        hidden_features=None,  # 隐藏层特征维度
        out_features=None,  # 输出特征维度
        act_layer=nn.GELU,  # 激活函数（默认GELU）
        drop=0.0,  # Dropout比率
    ):
        """
        初始化多层感知器
        
        参数：
        - in_features: 输入特征维度
        - hidden_features: 隐藏层特征维度（默认与输入维度相同）
        - out_features: 输出特征维度（默认与输入维度相同）
        - act_layer: 激活函数（默认GELU，平滑的ReLU变体）
        - drop: Dropout比率，防止过拟合
        """
        # 调用父类初始化方法
        super().__init__()
        
        # 设置输出和隐藏层维度，默认与输入维度相同
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 第一个全连接层：输入 -> 隐藏层
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 激活函数：增加非线性
        # GELU：平滑的ReLU变体，在Transformer中常用
        self.act = act_layer()
        
        # 第二个全连接层：隐藏层 -> 输出
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # Dropout层：随机"关闭"一部分神经元
        # 防止模型过度依赖特定特征
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播方法
        
        处理流程：
        1. 第一个全连接层
        2. 激活函数
        3. Dropout
        4. 第二个全连接层
        5. 再次Dropout
        
        参数：
        - x: 输入特征
        
        返回：
        - 变换后的特征
        """
        # 第一个全连接层
        x = self.fc1(x)
        
        # 激活函数：增加非线性
        x = self.act(x)
        
        # 第一次Dropout
        x = self.drop(x)
        
        # 第二个全连接层
        x = self.fc2(x)
        
        # 第二次Dropout
        x = self.drop(x)
        
        return x


class FeedForward1D(nn.Module):
    """
    一维前馈神经网络模块
    
    主要功能：
    1. 对一维特征进行非线性变换
    2. 使用全连接层和GELU激活
    3. 添加Dropout防止过拟合
    
    适用场景：
    - 序列数据处理
    - Transformer编码器中的前馈网络
    - 特征维度调整和非线性变换
    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        """
        初始化一维前馈网络
        
        参数：
        - dim: 输入和输出特征维度
        - hidden_dim: 隐藏层特征维度
        - dropout: Dropout比率
        """
        # 调用父类初始化方法
        super(FeedForward1D, self).__init__()
        
        # 使用顺序容器定义网络结构
        self.net = nn.Sequential(
            # 第一个全连接层：dim -> hidden_dim
            nn.Linear(dim, hidden_dim),
            
            # GELU激活函数：平滑的ReLU变体
            nn.GELU(),
            
            # 第一次Dropout
            nn.Dropout(dropout),
            
            # 第二个全连接层：hidden_dim -> dim
            nn.Linear(hidden_dim, dim),
            
            # 第二次Dropout
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        前向传播方法
        
        参数：
        - x: 输入特征
        
        返回：
        - 变换后的特征
        """
        return self.net(x)


class FeedForward2D(nn.Module):
    """
    二维前馈神经网络模块
    
    主要功能：
    1. 对二维特征图进行卷积变换
    2. 使用空洞卷积增大感受野
    3. 添加批归一化和LeakyReLU激活
    
    适用场景：
    - 图像特征处理
    - 深度伪造检测
    - 特征图增强和转换
    """
    def __init__(self, in_channel, out_channel):
        """
        初始化二维前馈网络
        
        参数：
        - in_channel: 输入特征图通道数
        - out_channel: 输出特征图通道数
        """
        # 调用父类初始化方法
        super(FeedForward2D, self).__init__()
        
        # 定义卷积网络结构
        self.conv = nn.Sequential(
            # 第一个卷积层：使用空洞卷积
            # 空洞卷积可以增大感受野，捕捉更大范围的特征
            nn.Conv2d(
                in_channel, out_channel, 
                kernel_size=3, padding=2, dilation=2
            ),
            
            # 批归一化：标准化特征图
            nn.BatchNorm2d(out_channel),
            
            # LeakyReLU激活：允许小的负值，防止梯度消失
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二个卷积层：进一步处理特征
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            
            # 再次批归一化
            nn.BatchNorm2d(out_channel),
            
            # 再次LeakyReLU激活
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        """
        前向传播方法
        
        参数：
        - x: 输入特征图
        
        返回：
        - 变换后的特征图
        """
        # 通过卷积网络处理特征图
        x = self.conv(x)
        
        return x
