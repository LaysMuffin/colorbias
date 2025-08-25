import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from collections import defaultdict
import argparse

# 导入模型
from improved_color_combination import (
    ImprovedColorCombinationHead, 
    BalancedColorCombinationDataset,
    IMPROVED_COLOR_COMBINATIONS
)

class CAMGenerator:
    """CAM (Class Activation Mapping) 生成器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedColorCombinationHead(num_classes=5).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 获取最后一个卷积层的特征
        self.features = None
        self.gradients = None
        
        # 注册钩子
        self._register_hooks()
        
        print(f"✅ 模型已加载: {model_path}")
        print(f"📱 使用设备: {self.device}")
    
    def _register_hooks(self):
        """注册钩子来获取特征和梯度"""
        def forward_hook(module, input, output):
            self.features = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # 获取最后一个卷积层（特征提取器的最后一层）
        last_conv_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
        
        if last_conv_layer is not None:
            last_conv_layer.register_forward_hook(forward_hook)
            last_conv_layer.register_backward_hook(backward_hook)
            print(f"✅ 已注册钩子到卷积层: {last_conv_layer}")
        else:
            print("❌ 未找到卷积层")
    
    def generate_cam(self, image, target_class=None):
        """生成CAM"""
        # 准备输入
        if isinstance(image, np.ndarray):
            # 如果是HWC格式，转换为CHW格式
            if image.shape[-1] == 3:  # HWC -> CHW
                image = np.transpose(image, (2, 0, 1))
            image = torch.FloatTensor(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            # 如果是HWC格式，转换为CHW格式
            if image.shape[-1] == 3:  # HWC -> CHW
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad = True
        
        # 前向传播
        outputs = self.model(image)
        logits = outputs['combination_logits']
        
        # 如果没有指定目标类别，使用预测的类别
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # 计算目标类别的得分
        score = logits[0, target_class]
        
        # 反向传播
        self.model.zero_grad()
        score.backward()
        
        # 获取特征和梯度
        if self.features is None or self.gradients is None:
            print("❌ 无法获取特征或梯度")
            return None
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # 生成CAM
        cam = torch.zeros(self.features.shape[2:], dtype=torch.float32, device=self.device)
        
        for i, w in enumerate(weights[0]):
            cam += w * self.features[0, i, :, :]
        
        # 应用ReLU
        cam = F.relu(cam)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, image, cam, save_path=None, alpha=0.6):
        """可视化CAM"""
        # 确保图像是numpy数组
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu().numpy()
            if image.shape[0] == 3:  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
        
        # 调整CAM大小以匹配图像
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # 创建热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # 转换图像格式
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        else:
            image = np.uint8(image)
        
        # 叠加
        cam_image = heatmap + alpha * np.float32(image) / 255
        cam_image = cam_image / np.max(cam_image)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # CAM热力图
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('CAM热力图')
        axes[1].axis('off')
        
        # 叠加结果
        axes[2].imshow(cam_image)
        axes[2].set_title('CAM叠加')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ CAM图片已保存: {save_path}")
        
        plt.show()
        
        return cam_image

def create_test_images():
    """创建测试图像"""
    test_images = []
    
    # 为每个颜色组合创建测试图像
    for combo_id in range(5):
        combo = IMPROVED_COLOR_COMBINATIONS[combo_id]
        print(f"创建 {combo['name']} 测试图像...")
        
        # 创建32x32的图像
        img = np.ones((32, 32, 3), dtype=np.uint8) * 255
        
        if combo_id == 0:  # 红白黑
            cv2.circle(img, (16, 16), 14, (0, 0, 255), 2)
            cv2.circle(img, (16, 16), 12, (255, 255, 255), -1)
            cv2.putText(img, '20', (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        elif combo_id == 1:  # 蓝白
            cv2.circle(img, (16, 16), 14, (255, 0, 0), -1)
            cv2.arrowedLine(img, (8, 16), (24, 16), (255, 255, 255), 2)
        elif combo_id == 2:  # 黄白黑
            cv2.circle(img, (16, 16), 14, (0, 255, 255), -1)
            cv2.rectangle(img, (10, 10), (22, 22), (0, 0, 0), 2)
        elif combo_id == 3:  # 红白
            cv2.circle(img, (16, 16), 14, (0, 0, 255), -1)
            cv2.putText(img, 'STOP', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        elif combo_id == 4:  # 黑白
            cv2.circle(img, (16, 16), 14, (0, 0, 0), 2)
            cv2.circle(img, (16, 16), 12, (255, 255, 255), -1)
        
        test_images.append({
            'image': img.astype(np.float32) / 255.0,
            'label': combo_id,
            'name': combo['name']
        })
    
    return test_images

def load_gtsrb_samples(gtsrb_path, num_samples=5):
    """加载GTSRB样本"""
    samples = []
    
    if not os.path.exists(gtsrb_path):
        print(f"❌ GTSRB路径不存在: {gtsrb_path}")
        return samples
    
    # 为每个颜色组合加载一个样本
    for combo_id in range(5):
        combo = IMPROVED_COLOR_COMBINATIONS[combo_id]
        classes = combo['classes']
        
        # 选择第一个类别
        class_id = classes[0]
        class_folder = os.path.join(gtsrb_path, 'Final_Training', 'Images', f'{class_id:05d}')
        
        if os.path.exists(class_folder):
            # 加载第一张图像
            for img_name in os.listdir(class_folder):
                if img_name.endswith('.ppm'):
                    img_path = os.path.join(class_folder, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((32, 32))
                        img_array = np.array(img).astype(np.float32) / 255.0
                        
                        samples.append({
                            'image': img_array,
                            'label': combo_id,
                            'name': f"{combo['name']} (GTSRB {class_id})"
                        })
                        break
                    except:
                        continue
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='生成CAM可视化')
    parser.add_argument('--model_path', default='best_improved_color_combination_head.pth', 
                       help='模型权重文件路径')
    parser.add_argument('--gtsrb_path', default='/home/hding22/color/GTSRB/GTSRB',
                       help='GTSRB数据集路径')
    parser.add_argument('--output_dir', default='cam_visualizations',
                       help='输出目录')
    parser.add_argument('--use_gtsrb', action='store_true',
                       help='使用GTSRB真实数据')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化CAM生成器
    cam_generator = CAMGenerator(args.model_path)
    
    # 准备测试数据
    if args.use_gtsrb:
        print("📸 加载GTSRB真实样本...")
        test_samples = load_gtsrb_samples(args.gtsrb_path)
    else:
        print("🎨 创建合成测试图像...")
        test_samples = create_test_images()
    
    print(f"✅ 准备生成 {len(test_samples)} 个样本的CAM可视化")
    
    # 生成CAM可视化
    for i, sample in enumerate(test_samples):
        print(f"\n🎯 处理样本 {i+1}/{len(test_samples)}: {sample['name']}")
        
        # 生成CAM
        cam = cam_generator.generate_cam(sample['image'], sample['label'])
        
        if cam is not None:
            # 保存路径
            save_path = os.path.join(args.output_dir, f'cam_{sample["name"]}.png')
            
            # 可视化
            cam_generator.visualize_cam(sample['image'], cam, save_path)
            
            # 保存原始CAM数据
            cam_data_path = os.path.join(args.output_dir, f'cam_{sample["name"]}.npy')
            np.save(cam_data_path, cam)
            print(f"💾 CAM数据已保存: {cam_data_path}")
        else:
            print(f"❌ 无法为 {sample['name']} 生成CAM")
    
    print(f"\n🎉 CAM可视化完成！结果保存在: {args.output_dir}")
    
    # 生成总结报告
    summary = {
        'model_path': args.model_path,
        'output_dir': args.output_dir,
        'num_samples': len(test_samples),
        'data_type': 'GTSRB' if args.use_gtsrb else 'Synthetic',
        'samples': [sample['name'] for sample in test_samples]
    }
    
    summary_path = os.path.join(args.output_dir, 'cam_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 总结报告已保存: {summary_path}")

if __name__ == "__main__":
    main()
