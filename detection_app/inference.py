import torch
from PIL import Image
import torchvision.transforms as transforms
import yaml
import os
from M2TR.models.m2tr import M2TR

class DeepfakeDetector:
    def __init__(self, model_path):
        # 加载配置文件
        with open('../configs/m2tr.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(f"\n使用设备: {self.device}")
        
        # 初始化模型
        print("\n正在初始化模型...")
        self.model = M2TR(config['MODEL'])
        
        # 打印模型结构
        # print("\n模型架构:")
        # print(self.model)
        
        # 加载模型权重
        print(f"\n正在加载权重文件: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取模型状态字典
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            print("找到模型状态字典")
        else:
            raise ValueError("权重文件中没有找到 'model_state' 键")

        # 获取当前模型的状态字典
        model_state = self.model.state_dict()
        print("\n模型架构与权重匹配检查:")
    
        # 检查形状匹配
        shape_mismatch = []
        missing_keys = []
        unexpected_keys = []
        
        # 检查权重文件中的键是否都在模型中存在，并且形状匹配
        for key, weight in state_dict.items():
            if key not in model_state:
                unexpected_keys.append(key)
                continue
            if weight.shape != model_state[key].shape:
                shape_mismatch.append(f"{key}: 权重形状 {weight.shape} vs 模型形状 {model_state[key].shape}")
        
        # 检查模型中的键是否都在权重文件中存在
        for key in model_state.keys():
            if key not in state_dict:
                missing_keys.append(key)
        
        # 打印检查结果
        if len(missing_keys) > 0:
            print("\n缺失的权重:")
            for key in missing_keys:
                print(f"- {key}")
        
        if len(unexpected_keys) > 0:
            print("\n多余的权重:")
            for key in unexpected_keys:
                print(f"- {key}")
        
        if len(shape_mismatch) > 0:
            print("\n形状不匹配的层:")
            for mismatch in shape_mismatch:
                print(f"- {mismatch}")
        
        if len(missing_keys) == 0 and len(unexpected_keys) == 0 and len(shape_mismatch) == 0:
            print("模型架构与权重完全匹配！")
        
        
        # 加载处理后的权重
        try:
            # missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            self.model.load_state_dict(state_dict)
            print("\n权重加载完成")
            # print(f"缺失的键: {missing_keys}")
            # print(f"未预期的键: {unexpected_keys}")
        except Exception as e:
            print("\n权重加载失败，错误信息:")
            print(e)
        
        # 将模型移至指定设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 设置图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),  # 与配置文件中的IMG_SIZE一致
            transforms.CenterCrop(320),     # 与MODEL.IMG_SIZE一致
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        print("\n初始化完成")
    
    def predict(self, image):
        # 预处理图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model({'img': image})
            logits = output['logits']
            
            # 打印原始输出
            print(f"\n原始logits: {logits}")
            
            # 使用softmax获取概率
            probs = torch.softmax(logits, dim=1)
            
            # 获取具体的预测概率
            real_prob = float(probs[0][0].cpu())
            fake_prob = float(probs[0][1].cpu())
            
            # 打印详细的预测信息
            print(f"预测概率 - 真实: {real_prob:.4f}, 伪造: {fake_prob:.4f}")
            
            is_fake = fake_prob > 0.5
            
        return is_fake, fake_prob