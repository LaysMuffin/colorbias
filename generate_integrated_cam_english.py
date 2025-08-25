#!/usr/bin/env python3
"""
集成颜色推理模型CAM可视化生成器 (英文版)
使用训练好的IntegratedColorReasoningModel生成类别激活映射
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib为英文
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class IntegratedCAMGenerator:
    """集成颜色推理模型CAM生成器"""
    
    def __init__(self, model_path: str = 'best_integrated_color_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'Red-White-Black (Speed Limit)',
            'Blue-White (Direction)',
            'Yellow-White-Black (Priority)',
            'Red-White (Stop)',
            'Black-White (Speed End)'
        ]
        self.color_combinations = [
            'Red-White-Black',
            'Blue-White', 
            'Yellow-White-Black',
            'Red-White',
            'Black-White'
        ]
        self.load_model()
        
    def load_model(self):
        """加载集成颜色推理模型"""
        try:
            from integrated_color_reasoning import IntegratedColorReasoningModel
            self.model = IntegratedColorReasoningModel(num_classes=5)
            
            # 加载预训练权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # 检查checkpoint格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 直接是状态字典
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Loaded integrated color reasoning model from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_test_image(self, color_combo_idx: int, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """生成测试图像"""
        height, width = size
        
        if color_combo_idx == 0:  # Red-White-Black
            # 红色背景，白色圆形，黑色数字
            img = np.ones((height, width, 3), dtype=np.uint8) * [255, 0, 0]  # Red background
            # 简单的几何形状
            img[height//4:3*height//4, width//4:3*width//4] = [255, 255, 255]  # White center
            img[height//3:2*height//3, width//3:2*width//3] = [0, 0, 0]  # Black inner
            
        elif color_combo_idx == 1:  # Blue-White
            # 蓝色背景，白色箭头
            img = np.ones((height, width, 3), dtype=np.uint8) * [0, 0, 255]  # Blue background
            # 白色三角形
            img[height//4:3*height//4, width//4:3*width//4] = [255, 255, 255]  # White center
            
        elif color_combo_idx == 2:  # Yellow-White-Black
            # 黄色背景，白色边框，黑色符号
            img = np.ones((height, width, 3), dtype=np.uint8) * [0, 255, 255]  # Yellow background
            img[10:height-10, 10:width-10] = [255, 255, 255]  # White border
            img[height//3:2*height//3, width//3:2*width//3] = [0, 0, 0]  # Black center
            
        elif color_combo_idx == 3:  # Red-White
            # 红色背景，白色八边形
            img = np.ones((height, width, 3), dtype=np.uint8) * [255, 0, 0]  # Red background
            # 白色中心区域
            img[height//4:3*height//4, width//4:3*width//4] = [255, 255, 255]  # White center
            
        else:  # Black-White
            # 白色背景，黑色符号
            img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
            img[20:height-20, 20:width-20] = [0, 0, 0]  # Black border
            img[height//2-5:height//2+5, :] = [0, 0, 0]  # Black line
            
        return img
    
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征"""
        # 转换为tensor
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # 通过颜色分类器提取特征
        with torch.no_grad():
            # 获取颜色分类器的特征
            if hasattr(self.model.color_classifier, 'extract_color_features'):
                color_features = self.model.color_classifier.extract_color_features(img_tensor)
            else:
                # 如果没有专门的特征提取方法，使用前向传播
                outputs = self.model.color_classifier(img_tensor)
                color_features = outputs['color_semantic']
            return color_features
    
    def generate_cam(self, image: np.ndarray, target_class: int) -> np.ndarray:
        """生成CAM热力图"""
        # 转换为tensor
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad_(True)
        
        # 获取目标类别的权重
        outputs = self.model(img_tensor)
        final_logits = outputs['final_logits']
        
        # 获取目标类别的梯度
        final_logits[0, target_class].backward()
        
        # 获取输入图像的梯度
        gradients = img_tensor.grad.clone()
        img_tensor.grad.zero_()
        
        # 计算权重
        weights = torch.mean(gradients, dim=(2, 3))
        
        # 生成CAM
        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * img_tensor, dim=1)
        cam = F.relu(cam)  # 只保留正值
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 调整大小到原始图像尺寸
        cam = F.interpolate(cam.unsqueeze(0), size=image.shape[:2], 
                          mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        return cam
    
    def create_visualization(self, image: np.ndarray, cam: np.ndarray, 
                           class_name: str, color_combo: str) -> np.ndarray:
        """创建可视化图像"""
        # 创建热力图
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加到原图
        overlay = heatmap * 0.7 + image * 0.3
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # 创建组合图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title(f'Original Image\n{color_combo}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 热力图
        axes[1].imshow(heatmap)
        axes[1].set_title('CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 叠加图
        axes[2].imshow(overlay)
        axes[2].set_title('CAM Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Integrated Color Reasoning CAM\n{class_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 转换为numpy数组
        fig.canvas.draw()
        try:
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        except AttributeError:
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_array = img_array[:, :, :3]  # 只保留RGB通道
        else:
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        
        return img_array
    
    def generate_all_cams(self, output_dir: str = 'integrated_cam_english'):
        """为所有颜色组合生成CAM"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎨 Generating CAM visualizations for integrated color reasoning model...")
        print(f"📁 Output directory: {output_dir}")
        
        results = []
        
        for i, (class_name, color_combo) in enumerate(zip(self.class_names, self.color_combinations)):
            print(f"  Processing {color_combo}...")
            
            # 生成测试图像
            image = self.generate_test_image(i)
            
            # 生成CAM
            cam = self.generate_cam(image, i)
            
            # 创建可视化
            vis_image = self.create_visualization(image, cam, class_name, color_combo)
            
            # 保存图像
            output_path = os.path.join(output_dir, f'integrated_cam_{i:02d}_{color_combo.lower().replace("-", "_")}.png')
            Image.fromarray(vis_image).save(output_path)
            
            # 保存CAM数据
            cam_data_path = os.path.join(output_dir, f'integrated_cam_{i:02d}_{color_combo.lower().replace("-", "_")}.npy')
            np.save(cam_data_path, cam)
            
            # 记录结果
            result = {
                'class_id': i,
                'class_name': class_name,
                'color_combo': color_combo,
                'image_path': output_path,
                'cam_data_path': cam_data_path,
                'cam_max': float(cam.max()),
                'cam_mean': float(cam.mean()),
                'cam_std': float(cam.std())
            }
            results.append(result)
            
            print(f"    ✅ Saved: {output_path}")
        
        # 保存分析报告
        analysis_path = os.path.join(output_dir, 'integrated_cam_analysis.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_info': {
                    'model_path': self.model_path,
                    'model_type': 'IntegratedColorReasoningModel',
                    'num_classes': 5
                },
                'generation_info': {
                    'total_images': len(results),
                    'output_directory': output_dir
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        # 创建总结图表
        self.create_summary_chart(results, output_dir)
        
        print(f"\n✅ Generated {len(results)} CAM visualizations")
        print(f"📊 Analysis saved to: {analysis_path}")
        
        return results
    
    def create_summary_chart(self, results: List[Dict], output_dir: str):
        """创建总结图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        class_names = [r['color_combo'] for r in results]
        cam_max = [r['cam_max'] for r in results]
        cam_mean = [r['cam_mean'] for r in results]
        cam_std = [r['cam_std'] for r in results]
        
        # CAM最大值
        bars1 = ax1.bar(range(len(class_names)), cam_max, color='skyblue', alpha=0.7)
        ax1.set_title('CAM Maximum Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Maximum CAM Value')
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, cam_max):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # CAM平均值
        bars2 = ax2.bar(range(len(class_names)), cam_mean, color='lightgreen', alpha=0.7)
        ax2.set_title('CAM Mean Values', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean CAM Value')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, cam_mean):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # CAM标准差
        bars3 = ax3.bar(range(len(class_names)), cam_std, color='salmon', alpha=0.7)
        ax3.set_title('CAM Standard Deviation', fontsize=12, fontweight='bold')
        ax3.set_ylabel('CAM Standard Deviation')
        ax3.set_xticks(range(len(class_names)))
        ax3.set_xticklabels(class_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, cam_std):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 综合统计
        ax4.scatter(cam_mean, cam_std, s=100, c=range(len(class_names)), 
                   cmap='viridis', alpha=0.7)
        for i, name in enumerate(class_names):
            ax4.annotate(name, (cam_mean[i], cam_std[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax4.set_xlabel('Mean CAM Value')
        ax4.set_ylabel('CAM Standard Deviation')
        ax4.set_title('CAM Statistics Overview', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Integrated Color Reasoning CAM Analysis Summary', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        summary_path = os.path.join(output_dir, 'integrated_cam_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary chart saved to: {summary_path}")


def main():
    """主函数"""
    print("🎨 Integrated Color Reasoning CAM Generator (English)")
    print("=" * 60)
    
    # 检查模型文件
    model_path = 'best_integrated_color_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run integrated_color_reasoning.py first to train the model.")
        return
    
    try:
        # 创建CAM生成器
        generator = IntegratedCAMGenerator(model_path)
        
        # 生成所有CAM
        results = generator.generate_all_cams()
        
        print("\n🎉 CAM generation completed successfully!")
        print(f"📁 Check the 'integrated_cam_english' directory for results")
        
    except Exception as e:
        print(f"❌ Error during CAM generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
