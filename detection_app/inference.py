import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import yaml
import os
from M2TR.models.m2tr import M2TR
import numpy as np
from typing import Union, Tuple, List
import cv2  # 添加 cv2 导入
import time

class DeepfakeDetector:
    def __init__(self, model_path):
        # 加载配置文件
        with open('../configs/m2tr.yaml', 'r', encoding='utf-8') as f:
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



    def add_scan_effect(self, image):
        """添加扫描动画效果"""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # 添加边框
        width, height = img.size
        border = 3
        draw.rectangle([(0, 0), (width-1, height-1)], outline='cyan', width=border)
        
        # 添加角标
        corner_length = 20
        # 左上角
        draw.line([(0, 0), (corner_length, 0)], fill='cyan', width=border)
        draw.line([(0, 0), (0, corner_length)], fill='cyan', width=border)
        # 右上角
        draw.line([(width-1, 0), (width-1-corner_length, 0)], fill='cyan', width=border)
        draw.line([(width-1, 0), (width-1, corner_length)], fill='cyan', width=border)
        # 左下角
        draw.line([(0, height-1), (corner_length, height-1)], fill='cyan', width=border)
        draw.line([(0, height-1), (0, height-1-corner_length)], fill='cyan', width=border)
        # 右下角
        draw.line([(width-1, height-1), (width-1-corner_length, height-1)], fill='cyan', width=border)
        draw.line([(width-1, height-1), (width-1, height-1-corner_length)], fill='cyan', width=border)
        
        return img


    def predict(self, input_data):
        """统一的预测接口，支持图片和视频输入"""
        # 处理视频输入
        if isinstance(input_data, str) and input_data.lower().endswith(('.mp4', '.avi', '.mov')):
            print("\n🎥 开始处理视频...")
            frames = self.extract_frames(input_data)
            frame_predictions = []
            fake_probs = []
            processed_frames = []  # 存储处理后的帧
            
            for i, frame in enumerate(frames):
                print(f"\n⏳ 正在分析第 {i+1}/{len(frames)} 帧...")
                # 添加扫描效果
                frame_with_effect = self.add_scan_effect(frame)
                processed_frames.append(frame_with_effect)
                
                # 对每一帧进行预测
                image = self.transform(frame).unsqueeze(0)
                image = image.to(self.device)
                
                with torch.no_grad():
                    print("🔍 运行深度伪造检测...")
                    output = self.model({'img': image})
                    logits = output['logits']
                    probs = torch.softmax(logits, dim=1)
                    fake_prob = float(probs[0][1].cpu())
                    print(f"📊 当前帧伪造概率: {fake_prob:.4f}")
                    fake_probs.append(fake_prob)

                time.sleep(0.1)  # 添加小延迟以展示进度
            
            # 计算平均伪造概率
            avg_fake_prob = np.mean(fake_probs)
            is_fake = avg_fake_prob > 0.5
            print(f"\n✨ 视频分析完成！平均伪造概率: {avg_fake_prob:.4f}")
            return is_fake, avg_fake_prob

        # 处理图片输入
        print("\n📸 开始处理图片...")
        if isinstance(input_data, str):
            input_data = Image.open(input_data).convert('RGB')
        # 添加扫描效果
        input_data_with_effect = self.add_scan_effect(input_data)
        print("🔄 预处理图像...")
        image = self.transform(input_data).unsqueeze(0)
        image = image.to(self.device)
        
        print("🔍 运行深度伪造检测...")
        with torch.no_grad():
            output = self.model({'img': image})
            logits = output['logits']
            print(f"\n原始logits: {logits}")
            probs = torch.softmax(logits, dim=1)
            real_prob = float(probs[0][0].cpu())
            fake_prob = float(probs[0][1].cpu())
            print(f"📊 检测结果 - 真实概率: {real_prob:.4f}, 伪造概率: {fake_prob:.4f}")
            is_fake = fake_prob > 0.5
            
        return is_fake, fake_prob, input_data_with_effect
    

    def extract_frames(self, video_path: str, num_frames: int = 30) -> List[Image.Image]:
        """从视频中均匀提取指定数量的帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
        
        cap.release()
        return frames
    

    # def predict_video(self, video_path: str, num_frames: int = 30) -> Tuple[bool, float, List[Tuple[bool, float]]]:
    #     """
    #     预测视频是否为深度伪造
    #     返回: (是否伪造, 伪造概率, 每帧的预测结果)
    #     """
    #     frames = self.extract_frames(video_path, num_frames)
    #     frame_predictions = []
    #     fake_probs = []

    #     for frame in frames:
    #         is_fake, fake_prob = self.predict(frame)
    #         frame_predictions.append((is_fake, fake_prob))
    #         fake_probs.append(fake_prob)

    #     # 计算平均伪造概率
    #     avg_fake_prob = np.mean(fake_probs)
    #     is_fake = avg_fake_prob > 0.5

    #     return is_fake, avg_fake_prob, frame_predictions