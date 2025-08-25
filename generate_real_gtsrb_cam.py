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
import random

# 导入模型
from improved_color_combination import (
    ImprovedColorCombinationHead, 
    BalancedColorCombinationDataset,
    IMPROVED_COLOR_COMBINATIONS
)

class RealGTSRBCAMGenerator:
    """真实GTSRB图片CAM生成器"""
    
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
        
        # 获取最后一个卷积层
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
    
    def visualize_cam(self, image, cam, save_path=None, alpha=0.6, title="CAM Visualization"):
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
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # CAM热力图
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('CAM Heatmap', fontsize=12)
        axes[1].axis('off')
        
        # 叠加结果
        axes[2].imshow(cam_image)
        axes[2].set_title('CAM Overlay', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ CAM图片已保存: {save_path}")
        
        plt.show()
        
        return cam_image

def load_gtsrb_samples_detailed(gtsrb_path, samples_per_combo=3):
    """加载详细的GTSRB样本"""
    samples = []
    
    if not os.path.exists(gtsrb_path):
        print(f"❌ GTSRB路径不存在: {gtsrb_path}")
        return samples
    
    # 为每个颜色组合加载多个样本
    for combo_id in range(5):
        combo = IMPROVED_COLOR_COMBINATIONS[combo_id]
        classes = combo['classes']
        
        print(f"📸 加载 {combo['name']} 组合的样本...")
        
        # 为每个类别加载样本
        for class_id in classes[:3]:  # 每个组合最多3个类别
            class_folder = os.path.join(gtsrb_path, 'Final_Training', 'Images', f'{class_id:05d}')
            
            if os.path.exists(class_folder):
                # 获取该类别下的所有图像
                image_files = [f for f in os.listdir(class_folder) if f.endswith('.ppm')]
                
                # 随机选择样本
                selected_files = random.sample(image_files, min(samples_per_combo, len(image_files)))
                
                for img_name in selected_files:
                    img_path = os.path.join(class_folder, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((32, 32))
                        img_array = np.array(img).astype(np.float32) / 255.0
                        
                        samples.append({
                            'image': img_array,
                            'label': combo_id,
                            'name': f"{combo['name']} (GTSRB {class_id})",
                            'class_id': class_id,
                            'combo_id': combo_id,
                            'file_name': img_name
                        })
                    except Exception as e:
                        print(f"❌ 加载图像失败 {img_path}: {e}")
                        continue
    
    return samples

def analyze_cam_results(cam_generator, samples, output_dir):
    """分析CAM结果"""
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n🎯 分析样本 {i+1}/{len(samples)}: {sample['name']}")
        
        # 生成CAM
        cam = cam_generator.generate_cam(sample['image'], sample['label'])
        
        if cam is not None:
            # 保存路径
            save_path = os.path.join(output_dir, f'cam_{sample["name"]}_{sample["file_name"]}.png')
            
            # 可视化
            cam_generator.visualize_cam(sample['image'], cam, save_path, 
                                      title=f"{sample['name']} - {sample['file_name']}")
            
            # 保存原始CAM数据
            cam_data_path = os.path.join(output_dir, f'cam_{sample["name"]}_{sample["file_name"]}.npy')
            np.save(cam_data_path, cam)
            
            # 分析CAM
            cam_analysis = analyze_single_cam(cam, sample)
            results.append(cam_analysis)
            
            print(f"💾 CAM数据已保存: {cam_data_path}")
        else:
            print(f"❌ 无法为 {sample['name']} 生成CAM")
    
    return results

def analyze_single_cam(cam, sample):
    """分析单个CAM"""
    # 计算CAM的统计信息
    cam_mean = np.mean(cam)
    cam_std = np.std(cam)
    cam_max = np.max(cam)
    cam_min = np.min(cam)
    
    # 计算注意力集中度（高值区域的比例）
    attention_threshold = 0.5
    high_attention_ratio = np.sum(cam > attention_threshold) / cam.size
    
    # 计算空间分布
    center_x, center_y = cam.shape[1] // 2, cam.shape[0] // 2
    center_attention = np.mean(cam[center_y-4:center_y+4, center_x-4:center_x+4])
    
    return {
        'sample_name': sample['name'],
        'class_id': sample['class_id'],
        'combo_id': sample['combo_id'],
        'file_name': sample['file_name'],
        'cam_stats': {
            'mean': float(cam_mean),
            'std': float(cam_std),
            'max': float(cam_max),
            'min': float(cam_min),
            'high_attention_ratio': float(high_attention_ratio),
            'center_attention': float(center_attention)
        }
    }

def main():
    parser = argparse.ArgumentParser(description='生成真实GTSRB图片的CAM可视化')
    parser.add_argument('--model_path', default='best_improved_color_combination_head.pth', 
                       help='模型权重文件路径')
    parser.add_argument('--gtsrb_path', default='/home/hding22/color/GTSRB/GTSRB',
                       help='GTSRB数据集路径')
    parser.add_argument('--output_dir', default='real_gtsrb_cam_detailed',
                       help='输出目录')
    parser.add_argument('--samples_per_combo', type=int, default=3,
                       help='每个颜色组合的样本数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    random.seed(42)
    
    # 初始化CAM生成器
    cam_generator = RealGTSRBCAMGenerator(args.model_path)
    
    # 加载真实GTSRB样本
    print("📸 加载真实GTSRB样本...")
    samples = load_gtsrb_samples_detailed(args.gtsrb_path, args.samples_per_combo)
    
    print(f"✅ 准备生成 {len(samples)} 个真实GTSRB样本的CAM可视化")
    
    # 分析CAM结果
    results = analyze_cam_results(cam_generator, samples, args.output_dir)
    
    print(f"\n🎉 真实GTSRB CAM可视化完成！结果保存在: {args.output_dir}")
    
    # 生成详细分析报告
    analysis_report = {
        'model_path': args.model_path,
        'output_dir': args.output_dir,
        'num_samples': len(samples),
        'data_type': 'Real GTSRB',
        'samples_per_combo': args.samples_per_combo,
        'cam_analysis': results,
        'summary_stats': generate_summary_stats(results)
    }
    
    # 保存分析报告
    report_path = os.path.join(args.output_dir, 'detailed_cam_analysis.json')
    with open(report_path, 'w') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 详细分析报告已保存: {report_path}")
    
    # 生成可视化总结
    generate_visualization_summary(results, args.output_dir)

def generate_summary_stats(results):
    """生成总结统计"""
    if not results:
        return {}
    
    # 按颜色组合分组
    combo_stats = defaultdict(list)
    for result in results:
        combo_id = result['combo_id']
        combo_stats[combo_id].append(result['cam_stats'])
    
    summary = {}
    for combo_id, stats_list in combo_stats.items():
        combo_name = IMPROVED_COLOR_COMBINATIONS[combo_id]['name']
        summary[combo_name] = {
            'num_samples': len(stats_list),
            'avg_attention_ratio': np.mean([s['high_attention_ratio'] for s in stats_list]),
            'avg_center_attention': np.mean([s['center_attention'] for s in stats_list]),
            'avg_cam_mean': np.mean([s['mean'] for s in stats_list]),
            'avg_cam_std': np.mean([s['std'] for s in stats_list])
        }
    
    return summary

def generate_visualization_summary(results, output_dir):
    """生成可视化总结"""
    if not results:
        return
    
    # 创建总结图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 按颜色组合分组的注意力比例
    combo_attention = defaultdict(list)
    for result in results:
        combo_name = IMPROVED_COLOR_COMBINATIONS[result['combo_id']]['name']
        combo_attention[combo_name].append(result['cam_stats']['high_attention_ratio'])
    
    # 绘制注意力比例箱线图
    attention_data = [combo_attention[name] for name in combo_attention.keys()]
    axes[0, 0].boxplot(attention_data, labels=list(combo_attention.keys()))
    axes[0, 0].set_title('Attention Ratio by Color Combination')
    axes[0, 0].set_ylabel('High Attention Ratio')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 绘制中心注意力分布
    center_attention = [r['cam_stats']['center_attention'] for r in results]
    axes[0, 1].hist(center_attention, bins=20, alpha=0.7)
    axes[0, 1].set_title('Center Attention Distribution')
    axes[0, 1].set_xlabel('Center Attention')
    axes[0, 1].set_ylabel('Frequency')
    
    # 绘制CAM均值分布
    cam_means = [r['cam_stats']['mean'] for r in results]
    axes[1, 0].hist(cam_means, bins=20, alpha=0.7, color='green')
    axes[1, 0].set_title('CAM Mean Distribution')
    axes[1, 0].set_xlabel('CAM Mean')
    axes[1, 0].set_ylabel('Frequency')
    
    # 绘制CAM标准差分布
    cam_stds = [r['cam_stats']['std'] for r in results]
    axes[1, 1].hist(cam_stds, bins=20, alpha=0.7, color='red')
    axes[1, 1].set_title('CAM Standard Deviation Distribution')
    axes[1, 1].set_xlabel('CAM Standard Deviation')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # 保存总结图表
    summary_path = os.path.join(output_dir, 'cam_analysis_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"📊 分析总结图表已保存: {summary_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
