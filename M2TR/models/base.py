# 导入PyTorch深度学习相关模块
import torch.nn as nn


class BaseNetwork(nn.Module):
    """
    基础神经网络类 - 提供通用的网络实用方法
    
    主要功能：
    1. 提供网络参数打印方法
    2. 提供权重初始化方法
    3. 作为所有自定义神经网络模型的基类
    
    适用场景：
    - 深度学习模型开发
    - 网络结构分析和调试
    - 统一权重初始化策略
    """
    def __init__(self):
        """
        初始化基础网络
        
        调用父类（nn.Module）的初始化方法
        为后续的网络模型提供基础
        """
        # 调用父类初始化方法
        super(BaseNetwork, self).__init__()

    def print_network(self):
        """
        打印网络参数信息
        
        功能：
        1. 计算网络总参数数量
        2. 打印网络类型和参数量
        
        特点：
        - 支持单个网络和网络列表
        - 参数量以百万为单位显示
        - 提供快速了解网络规模的方法
        """
        # 处理网络列表的情况
        if isinstance(self, list):
            self = self[0]
        
        # 初始化参数计数器
        num_params = 0
        
        # 遍历所有网络参数并计数
        for param in self.parameters():
            # 累加参数数量（numel()返回张量中元素总数）
            num_params += param.numel()
        
        # 打印网络信息
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).'
            % (type(self).__name__, num_params / 1000000)
        )

    def init_weights(self, init_type='normal', gain=0.02):
        """
        初始化网络权重
        
        目的：为神经网络设置合适的初始权重，提高训练稳定性和收敛速度
        
        参数：
        - init_type: 权重初始化方法
            'normal': 正态分布初始化
            'xavier': Xavier正态分布初始化
            'xavier_uniform': Xavier均匀分布初始化
            'kaiming': Kaiming正态分布初始化
            'orthogonal': 正交矩阵初始化
            'none': 使用PyTorch默认初始化
        
        - gain: 初始化强度调节参数
        
        支持的初始化层：
        - 卷积层（Conv）
        - 全连接层（Linear）
        - 实例归一化层（InstanceNorm2d）
        """
        def init_func(m):
            """
            内部权重初始化函数
            
            针对不同类型的网络层应用不同的初始化策略
            """
            # 获取层的类名
            classname = m.__class__.__name__
            
            # 处理实例归一化层
            if classname.find('InstanceNorm2d') != -1:
                # 权重初始化为1.0
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                
                # 偏置初始化为0.0
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            
            # 处理卷积层和全连接层
            elif hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1
            ):
                # 根据指定的初始化类型选择初始化方法
                if init_type == 'normal':
                    # 正态分布初始化
                    nn.init.normal_(m.weight.data, 0.0, gain)
                
                elif init_type == 'xavier':
                    # Xavier正态分布初始化
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                
                elif init_type == 'xavier_uniform':
                    # Xavier均匀分布初始化
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                
                elif init_type == 'kaiming':
                    # Kaiming正态分布初始化（适合ReLU）
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
                elif init_type == 'orthogonal':
                    # 正交矩阵初始化
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                
                elif init_type == 'none':
                    # 使用PyTorch默认初始化方法
                    m.reset_parameters()
                
                else:
                    # 不支持的初始化方法
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented'
                        % init_type
                    )
                
                # 偏置项始终初始化为0
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        # 对整个网络应用初始化函数
        self.apply(init_func)

        # 递归处理子模块
        for m in self.children():
            # 如果子模块有自定义权重初始化方法，调用该方法
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
