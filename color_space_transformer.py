# color_space_transformer.py
# 颜色空间转换模块 - 提供颜色感知特征

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple
import math

class TrueColorSpaceConverter(nn.Module):
    """真正的颜色空间转换器 - 基于数学公式"""
    
    def __init__(self):
        super().__init__()
        
        # 预定义的转换矩阵 - 注册为缓冲区，会自动移动到正确设备
        self.register_buffer('rgb_to_xyz_matrix', torch.tensor([
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505]
        ], dtype=torch.float32))
        
        self.register_buffer('xyz_to_lab_matrix', torch.tensor([
            [0.0, 116.0, 0.0],
            [500.0, -500.0, 0.0],
            [0.0, 200.0, -200.0]
        ], dtype=torch.float32))
    
    def rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """RGB到HSV的真正转换"""
        # 归一化到[0,1]
        rgb_norm = torch.clamp(rgb, 0, 1)
        
        # 计算最大值和最小值
        max_rgb, _ = torch.max(rgb_norm, dim=1, keepdim=True)
        min_rgb, _ = torch.min(rgb_norm, dim=1, keepdim=True)
        diff = max_rgb - min_rgb
        
        # 计算饱和度
        saturation = torch.where(max_rgb > 0, diff / max_rgb, torch.zeros_like(diff))
        
        # 计算色调
        hue = torch.zeros_like(rgb_norm[:, :1, :, :])
        
        # R通道最大
        r_mask = (rgb_norm[:, 0:1, :, :] == max_rgb) & (diff > 0)
        hue[r_mask] = (60 * ((rgb_norm[:, 1:2, :, :] - rgb_norm[:, 2:3, :, :]) / diff))[r_mask] % 360
        
        # G通道最大
        g_mask = (rgb_norm[:, 1:2, :, :] == max_rgb) & (diff > 0)
        hue[g_mask] = (60 * ((rgb_norm[:, 2:3, :, :] - rgb_norm[:, 0:1, :, :]) / diff + 2))[g_mask] % 360
        
        # B通道最大
        b_mask = (rgb_norm[:, 2:3, :, :] == max_rgb) & (diff > 0)
        hue[b_mask] = (60 * ((rgb_norm[:, 0:1, :, :] - rgb_norm[:, 1:2, :, :]) / diff + 4))[b_mask] % 360
        
        # 归一化色调到[0,1]
        hue = hue / 360.0
        
        return torch.cat([hue, saturation, max_rgb], dim=1)
    
    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """RGB到Lab的真正转换"""
        # 归一化到[0,1]
        rgb_norm = torch.clamp(rgb, 0, 1)
        
        # 应用gamma校正
        rgb_gamma = torch.where(rgb_norm > 0.04045, 
                               torch.pow((rgb_norm + 0.055) / 1.055, 2.4),
                               rgb_norm / 12.92)
        
        # RGB到XYZ转换
        batch_size, _, height, width = rgb_gamma.shape
        rgb_flat = rgb_gamma.view(batch_size, 3, -1)
        
        # 矩阵乘法
        xyz_flat = torch.bmm(self.rgb_to_xyz_matrix.unsqueeze(0).expand(batch_size, -1, -1), rgb_flat)
        xyz = xyz_flat.view(batch_size, 3, height, width)
        
        # 归一化XYZ
        xyz_norm = xyz / torch.tensor([0.9505, 1.0, 1.0890], device=xyz.device).view(1, 3, 1, 1)
        
        # XYZ到Lab转换
        xyz_flat = xyz_norm.view(batch_size, 3, -1)
        lab_flat = torch.bmm(self.xyz_to_lab_matrix.unsqueeze(0).expand(batch_size, -1, -1), xyz_flat)
        lab = lab_flat.view(batch_size, 3, height, width)
        
        # 应用非线性变换
        lab[:, 0, :, :] = 116 * torch.pow(xyz_norm[:, 1, :, :], 1/3) - 16  # L
        lab[:, 1, :, :] = 500 * (torch.pow(xyz_norm[:, 0, :, :], 1/3) - torch.pow(xyz_norm[:, 1, :, :], 1/3))  # a
        lab[:, 2, :, :] = 200 * (torch.pow(xyz_norm[:, 1, :, :], 1/3) - torch.pow(xyz_norm[:, 2, :, :], 1/3))  # b
        
        return lab

class ColorInvarianceFeatures(nn.Module):
    """颜色不变性特征提取器"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        # 灰度特征提取
        self.grayscale_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        
        # 边缘特征提取
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        
        # 纹理特征提取
        self.texture_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(24, 16, 1),  # 8*3 = 24
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 1),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 灰度特征
        grayscale = self.grayscale_extractor(x)
        
        # 边缘特征
        edges = self.edge_extractor(x)
        
        # 纹理特征
        texture = self.texture_extractor(x)
        
        # 融合特征
        combined = torch.cat([grayscale, edges, texture], dim=1)
        fused = self.feature_fusion(combined)
        
        return {
            'grayscale': grayscale,
            'edges': edges,
            'texture': texture,
            'fused': fused,
            'combined': combined
        }

class MultiScaleColorAnalysis(nn.Module):
    """多尺度颜色分析"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        # 不同尺度的卷积核
        self.scale1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(input_channels, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(input_channels, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # 注意力机制
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(48, 16, 1),  # 16*3 = 48
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),  # 3个尺度
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(48, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 多尺度特征
        scale1_feat = self.scale1(x)
        scale2_feat = self.scale2(x)
        scale3_feat = self.scale3(x)
        
        # 拼接所有尺度特征
        all_scales = torch.cat([scale1_feat, scale2_feat, scale3_feat], dim=1)
        
        # 计算注意力权重
        attention_weights = self.scale_attention(all_scales)
        
        # 加权融合
        weighted_features = []
        for i in range(3):
            start_idx = i * 16
            end_idx = (i + 1) * 16
            weighted = all_scales[:, start_idx:end_idx, :, :] * attention_weights[:, i:i+1, :, :]
            weighted_features.append(weighted)
        
        weighted_combined = torch.cat(weighted_features, dim=1)
        
        # 最终融合
        fused = self.fusion(weighted_combined)
        
        return {
            'scale1': scale1_feat,
            'scale2': scale2_feat,
            'scale3': scale3_feat,
            'attention_weights': attention_weights,
            'weighted_features': weighted_combined,
            'fused': fused
        }

class ColorSemanticEncoder(nn.Module):
    """颜色语义编码器 - 学习颜色语义表示"""
    
    def __init__(self, input_channels=3, num_colors=7):
        super().__init__()
        self.num_colors = num_colors
        
        # 颜色语义特征提取
        self.semantic_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 颜色分类器
        self.color_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_colors),
            nn.Softmax(dim=1)
        )
        
        # 语义嵌入
        self.semantic_embedding = nn.Embedding(num_colors, 16)
        
        # 语义融合
        self.semantic_fusion = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 1),  # 特征 + 语义嵌入
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 语义特征提取
        semantic_features = self.semantic_extractor(x)
        
        # 颜色分类
        color_probs = self.color_classifier(semantic_features)
        color_indices = torch.argmax(color_probs, dim=1)
        
        # 语义嵌入
        semantic_embeddings = self.semantic_embedding(color_indices)  # [B, 16]
        
        # 扩展语义嵌入到空间维度
        batch_size, _, height, width = semantic_features.shape
        semantic_embeddings_spatial = semantic_embeddings.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        
        # 融合特征和语义
        combined = torch.cat([semantic_features, semantic_embeddings_spatial], dim=1)
        fused = self.semantic_fusion(combined)
        
        return {
            'semantic_features': semantic_features,
            'color_probs': color_probs,
            'color_indices': color_indices,
            'semantic_embeddings': semantic_embeddings,
            'fused': fused
        }

class ColorSpaceTransformer(nn.Module):
    """增强的颜色空间转换模块 - 整合所有颜色感知功能"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        self.input_channels = input_channels
        
        # 真正的颜色空间转换器
        self.true_converter = TrueColorSpaceConverter()
        
        # 颜色不变性特征
        self.invariance_features = ColorInvarianceFeatures(input_channels)
        
        # 多尺度颜色分析
        self.multi_scale_analysis = MultiScaleColorAnalysis(input_channels)
        
        # 颜色语义编码器
        self.semantic_encoder = ColorSemanticEncoder(input_channels)
        
        # 颜色通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, input_channels, 1),
            nn.Sigmoid()
        )
        
        # 全局颜色空间融合
        # 计算总通道数: RGB(3) + HSV(3) + Lab(3) + 对手色(3) + 不变性(8) + 多尺度(16) + 语义(16) = 52
        total_channels = input_channels * 4 + 8 + 16 + 16
        self.global_fusion = nn.Sequential(
            nn.Conv2d(total_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 1)
        )
        
        # 颜色一致性损失权重
        self.color_consistency_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播 - 生成增强的颜色感知特征"""
        batch_size = x.size(0)
        
        # 原始RGB特征
        rgb_features = x
        
        # 真正的颜色空间转换
        hsv_features = self.true_converter.rgb_to_hsv(x)
        lab_features = self.true_converter.rgb_to_lab(x)
        
        # 对手色空间 (R-G, R+G-2B, R+G+B)
        opponent_features = torch.stack([
            x[:, 0, :, :] - x[:, 1, :, :],  # R-G
            x[:, 0, :, :] + x[:, 1, :, :] - 2 * x[:, 2, :, :],  # R+G-2B
            x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]  # R+G+B
        ], dim=1)
        
        # 颜色不变性特征
        invariance_outputs = self.invariance_features(x)
        
        # 多尺度颜色分析
        multi_scale_outputs = self.multi_scale_analysis(x)
        
        # 颜色语义编码
        semantic_outputs = self.semantic_encoder(x)
        
        # 颜色通道注意力
        channel_weights = self.channel_attention(x)
        attended_rgb = rgb_features * channel_weights
        
        # 全局颜色空间融合
        all_color_spaces = torch.cat([
            attended_rgb,           # RGB
            hsv_features,           # HSV
            lab_features,           # Lab
            opponent_features,      # 对手色
            invariance_outputs['fused'],      # 不变性特征
            multi_scale_outputs['fused'],     # 多尺度特征
            semantic_outputs['fused']         # 语义特征
        ], dim=1)
        
        fused_features = self.global_fusion(all_color_spaces)
        
        return {
            'rgb': attended_rgb,
            'hsv': hsv_features,
            'lab': lab_features,
            'opponent': opponent_features,
            'invariance': invariance_outputs,
            'multi_scale': multi_scale_outputs,
            'semantic': semantic_outputs,
            'channel_weights': channel_weights,
            'fused': fused_features,
            'all_spaces': all_color_spaces
        }
    
    def compute_color_consistency_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算颜色一致性损失"""
        # 计算不同颜色空间之间的相关性
        hsv_flat = outputs['hsv'].view(outputs['hsv'].size(0), -1)
        lab_flat = outputs['lab'].view(outputs['lab'].size(0), -1)
        
        # 归一化
        hsv_norm = F.normalize(hsv_flat, dim=1)
        lab_norm = F.normalize(lab_flat, dim=1)
        
        # 计算相关性
        correlation = torch.mm(hsv_norm, lab_norm.t())
        
        # 惩罚高相关性（鼓励多样性）
        consistency_loss = torch.mean(torch.abs(correlation))
        
        return consistency_loss * self.color_consistency_weight

def test_color_space_transformer():
    """测试增强的颜色空间转换模块"""
    print("🎨 测试增强的颜色空间转换模块")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 4
    channels = 3
    height, width = 32, 32
    
    test_images = torch.randn(batch_size, channels, height, width).to(device)
    print(f"测试图像形状: {test_images.shape}")
    
    # 测试增强的颜色空间转换器
    color_transformer = ColorSpaceTransformer().to(device)
    color_outputs = color_transformer(test_images)
    
    print(f"\n📊 增强颜色空间转换器输出:")
    for key, value in color_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {type(value)}")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"    {sub_key}: {sub_value.shape}")
    
    # 测试颜色一致性损失
    consistency_loss = color_transformer.compute_color_consistency_loss(color_outputs)
    print(f"\n🎯 颜色一致性损失: {consistency_loss.item():.4f}")
    
    print(f"\n✅ 增强颜色空间转换模块测试完成")

if __name__ == '__main__':
    test_color_space_transformer()
