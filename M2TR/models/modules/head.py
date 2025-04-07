# 导入PyTorch深度学习相关模块
import torch.nn as nn

# 导入自定义的反卷积模块
from M2TR.models.modules.conv_block import Deconv


class Classifier2D(nn.Module):
    """
    二维分类器模块 - 将特征映射到类别概率

    主要功能：
    1. 对输入特征进行线性投影
    2. 将特征转换为类别预测
    3. 根据需要添加dropout和激活函数

    适用场景：
    - 图像分类
    - 深度伪造检测
    - 多类别预测任务
    """
    def __init__(
        self,
        dim_in,  # 输入特征的维度
        num_classes,  # 需要分类的类别数量
        dropout_rate=0.0,  # 随机失活比率
        act_func="softmax",  # 激活函数类型
    ):
        """
        初始化分类器

        参数：
        - dim_in: 输入特征的维度（通道数）
        - num_classes: 分类的类别数
        - dropout_rate: 随机失活比率（防止过拟合）
        - act_func: 激活函数类型（'softmax'或'sigmoid'）
        """
        # 调用父类初始化方法
        super(Classifier2D, self).__init__()

        # 如果dropout_rate大于0，添加随机失活层
        # 随机失活可以防止模型过度依赖某些特定特征，提高泛化能力
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # 线性投影层：将输入特征转换为类别预测
        # 例如：将512维特征转换为2个类别的预测
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # 根据指定的激活函数选择输出处理方式
        if act_func == "softmax":
            # Softmax：用于多类别分类，输出每个类别的概率和为1
            # 适合互斥的类别预测（如真实/伪造）
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            # Sigmoid：用于二分类或多标签分类
            # 每个类别的概率独立计算，可以大于0小于1
            self.act = nn.Sigmoid()
        else:
            # 不支持的激活函数，抛出异常
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        """
        前向传播方法

        处理流程：
        1. 可选：应用dropout
        2. 线性投影到类别空间
        3. 在推理阶段应用激活函数

        参数：
        - x: 输入特征

        返回：
        - 类别预测
        """
        # 如果存在dropout，应用随机失活
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        # 线性投影：将特征转换为类别预测
        x = self.projection(x)

        # 仅在非训练阶段（推理/测试）应用激活函数
        # 训练阶段通常使用交叉熵损失，不需要额外激活
        if not self.training:
            x = self.act(x)

        return x


class Localizer(nn.Module):
    """
    图像定位器模块 - 生成特征图或热力图

    主要功能：
    1. 通过反卷积和卷积层逐步处理特征
    2. 生成与输入尺寸相关的输出特征图
    3. 使用Sigmoid将输出映射到0-1范围

    适用场景：
    - 图像分割
    - 区域突出显示
    - 深度伪造区域定位
    """
    def __init__(self, in_channel, output_channel):
        """
        初始化定位器

        参数：
        - in_channel: 输入特征通道数
        - output_channel: 输出特征通道数
        """
        # 调用父类初始化方法
        super(Localizer, self).__init__()

        # 第一个反卷积层：上采样和特征转换
        # 使用Deconv模块放大特征图
        self.deconv1 = Deconv(in_channel, in_channel)

        # 中间特征维度：减半输入通道数
        hidden_dim = in_channel // 2

        # 第一个卷积层：调整特征通道并添加非线性
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 第二个反卷积层：继续处理特征
        self.deconv2 = Deconv(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # 第二个卷积层：生成最终输出
        # 包含LeakyReLU和卷积
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                hidden_dim, output_channel, kernel_size=3, stride=1, padding=1
            ),
        )

        # Sigmoid激活：将输出映射到0-1范围
        # 适合生成概率图或热力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播方法

        处理流程：
        1. 第一次反卷积和卷积
        2. 第二次反卷积和卷积
        3. Sigmoid激活

        参数：
        - x: 输入特征

        返回：
        - 0-1范围的特征图
        """
        # 第一次反卷积：上采样
        out = self.deconv1(x)

        # 第一次卷积：调整特征
        out = self.conv1(out)

        # 第二次反卷积：继续处理
        out = self.deconv2(out)

        # 第二次卷积：生成最终输出
        out = self.conv2(out)

        # Sigmoid激活：映射到0-1
        return self.sigmoid(out)
