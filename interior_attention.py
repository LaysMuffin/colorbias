# interior_attention.py
# Interior-Only注意力机制 - 专注标志内部区域

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
import cv2

class InteriorAttention(nn.Module):
    """Interior-Only注意力机制 - 专注标志内部区域，排除边框和背景"""
    
    def __init__(self, input_channels=3, attention_dim=64):
        super().__init__()
        self.input_channels = input_channels
        self.attention_dim = attention_dim
        
        # 边缘检测器 - 用于识别边框
        self.edge_detector = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # 形状检测器 - 用于识别形状边界
        self.shape_detector = nn.Sequential(
            nn.Conv2d(input_channels, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # 内部区域预测器
        self.interior_predictor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # 注意力融合网络
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3个注意力图
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # 内部特征提取器
        self.interior_feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        
        # 形状抑制器 - 抑制形状信息
        self.shape_suppressor = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.Tanh()  # 输出范围[-1,1]，用于抑制
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 - 生成内部注意力掩码和特征"""
        batch_size = x.size(0)
        
        # 1. 边缘检测 - 识别边框
        edge_map = self.edge_detector(x)
        
        # 2. 形状检测 - 识别形状边界
        shape_map = self.shape_detector(x)
        
        # 3. 内部区域预测 - 预测真正的内部区域
        interior_map = self.interior_predictor(x)
        
        # 4. 注意力融合 - 结合三种注意力图
        attention_inputs = torch.cat([edge_map, shape_map, interior_map], dim=1)
        fused_attention = self.attention_fusion(attention_inputs)
        
        # 5. 生成内部注意力掩码 - 排除边框和形状边界
        # 边缘和形状区域应该被抑制
        edge_suppression = 1.0 - edge_map
        shape_suppression = 1.0 - shape_map
        
        # 内部注意力掩码 = 内部区域 * 边缘抑制 * 形状抑制
        interior_attention_mask = interior_map * edge_suppression * shape_suppression
        
        # 6. 提取内部特征
        interior_features = self.interior_feature_extractor(x)
        
        # 7. 形状抑制 - 抑制形状相关的特征
        shape_suppressed_features = self.shape_suppressor(interior_features)
        
        # 8. 应用注意力掩码到特征
        attended_features = interior_features * interior_attention_mask
        
        return attended_features, interior_attention_mask, edge_map, shape_map
    
    def compute_interior_attention_loss(self, attention_mask: torch.Tensor, 
                                      logits: torch.Tensor, 
                                      targets: torch.Tensor) -> torch.Tensor:
        """计算内部注意力损失"""
        batch_size = attention_mask.size(0)
        
        # 计算注意力强度
        attention_strength = torch.mean(attention_mask, dim=[2, 3])  # [B, 1]
        
        # 颜色相关类别需要高内部注意力
        color_related_classes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        
        # 创建目标掩码
        target_mask = torch.zeros_like(attention_strength)
        for i in range(batch_size):
            if targets[i].item() in color_related_classes:
                target_mask[i] = 0.8  # 颜色相关类别需要高注意力
            else:
                target_mask[i] = 0.3  # 其他类别需要较低注意力
        
        # 计算注意力损失
        attention_loss = F.mse_loss(attention_strength, target_mask)
        
        return attention_loss
    
    def compute_shape_suppression_loss(self, features: torch.Tensor, 
                                     attention_mask: torch.Tensor) -> torch.Tensor:
        """计算形状抑制损失"""
        # 计算特征的空间梯度（形状信息）
        grad_x = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
        grad_y = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        
        # 调整梯度尺寸以匹配
        grad_x_padded = F.pad(grad_x, (0, 1, 0, 0))  # 在宽度维度填充
        grad_y_padded = F.pad(grad_y, (0, 0, 0, 1))  # 在高度维度填充
        
        shape_info = grad_x_padded + grad_y_padded
        
        # 在内部区域，形状信息应该被抑制
        interior_shape_info = shape_info * attention_mask
        
        # 惩罚高形状信息
        suppression_loss = torch.mean(interior_shape_info)
        
        return suppression_loss

class AdaptiveInteriorAttention(nn.Module):
    """自适应内部注意力 - 根据类别动态调整注意力策略"""
    
    def __init__(self, input_channels=3, num_classes=43):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 基础内部注意力
        self.base_interior_attention = InteriorAttention(input_channels)
        
        # 类别特定的注意力权重
        self.class_attention_weights = nn.Parameter(torch.ones(num_classes))
        
        # 自适应融合网络
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(num_classes + 1, 32),  # 类别 + 基础注意力强度
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, 16)
        
        # 类别感知特征提取
        self.class_aware_extractor = nn.Sequential(
            nn.Conv2d(input_channels + 16, 32, 3, padding=1),  # 图像 + 类别嵌入
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = x.size(0)
        
        # 基础内部注意力
        interior_features, attention_mask, edge_map, shape_map = self.base_interior_attention(x)
        
        # 计算基础注意力强度
        base_attention_strength = torch.mean(attention_mask, dim=[2, 3])
        
        if targets is not None:
            # 类别特定的注意力调整
            class_weights = self.class_attention_weights[targets]  # [B]
            
            # 创建类别one-hot编码
            class_onehot = F.one_hot(targets, num_classes=self.num_classes).float()  # [B, num_classes]
            
            # 自适应融合
            fusion_input = torch.cat([class_onehot, base_attention_strength], dim=1)
            adaptive_weight = self.adaptive_fusion(fusion_input)  # [B, 1]
            
            # 类别嵌入
            class_embeddings = self.class_embedding(targets)  # [B, 16]
            
            # 扩展类别嵌入到空间维度
            _, _, height, width = x.shape
            class_embeddings_spatial = class_embeddings.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
            
            # 类别感知特征提取
            class_aware_input = torch.cat([x, class_embeddings_spatial], dim=1)
            class_aware_features = self.class_aware_extractor(class_aware_input)
            
            # 自适应注意力掩码
            adaptive_attention_mask = attention_mask * adaptive_weight.unsqueeze(-1).unsqueeze(-1)
            
            return {
                'interior_features': interior_features,
                'class_aware_features': class_aware_features,
                'attention_mask': adaptive_attention_mask,
                'base_attention_mask': attention_mask,
                'edge_map': edge_map,
                'shape_map': shape_map,
                'adaptive_weight': adaptive_weight,
                'class_weights': class_weights,
                'base_attention_strength': base_attention_strength
            }
        else:
            return {
                'interior_features': interior_features,
                'attention_mask': attention_mask,
                'edge_map': edge_map,
                'shape_map': shape_map,
                'base_attention_strength': base_attention_strength
            }
    
    def compute_adaptive_attention_loss(self, outputs: Dict[str, torch.Tensor], 
                                      targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算自适应注意力损失"""
        # 基础内部注意力损失
        base_attention_loss = self.base_interior_attention.compute_interior_attention_loss(
            outputs['attention_mask'], None, targets
        )
        
        # 形状抑制损失
        shape_suppression_loss = self.base_interior_attention.compute_shape_suppression_loss(
            outputs['interior_features'], outputs['attention_mask']
        )
        
        # 类别一致性损失 - 确保同类别的注意力模式相似
        class_attention_strengths = outputs['base_attention_strength']
        class_consistency_loss = 0.0
        
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                if targets[i] == targets[j]:
                    # 同类别的注意力强度应该相似
                    consistency = torch.abs(class_attention_strengths[i] - class_attention_strengths[j])
                    class_consistency_loss += consistency
        
        if len(targets) > 1:
            class_consistency_loss = class_consistency_loss / (len(targets) * (len(targets) - 1) / 2)
        
        # 自适应权重正则化
        adaptive_weight_regularization = torch.mean(outputs['adaptive_weight'])
        
        return {
            'base_attention_loss': base_attention_loss,
            'shape_suppression_loss': shape_suppression_loss,
            'class_consistency_loss': class_consistency_loss,
            'adaptive_weight_regularization': adaptive_weight_regularization,
            'total_loss': base_attention_loss + 0.1 * shape_suppression_loss + 
                         0.05 * class_consistency_loss + 0.01 * adaptive_weight_regularization
        }

def test_interior_attention():
    """测试Interior-Only注意力机制"""
    print("🎯 测试Interior-Only注意力机制")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 4
    channels = 3
    height, width = 32, 32
    
    test_images = torch.randn(batch_size, channels, height, width).to(device)
    test_targets = torch.randint(0, 43, (batch_size,)).to(device)
    
    print(f"测试图像形状: {test_images.shape}")
    print(f"测试目标形状: {test_targets.shape}")
    
    # 测试基础内部注意力
    print(f"\n🎯 测试基础内部注意力:")
    interior_attention = InteriorAttention().to(device)
    interior_features, attention_mask, edge_map, shape_map = interior_attention(test_images)
    
    print(f"  内部特征形状: {interior_features.shape}")
    print(f"  注意力掩码形状: {attention_mask.shape}")
    print(f"  边缘图形状: {edge_map.shape}")
    print(f"  形状图形状: {shape_map.shape}")
    
    # 测试注意力损失
    attention_loss = interior_attention.compute_interior_attention_loss(
        attention_mask, None, test_targets
    )
    print(f"  注意力损失: {attention_loss.item():.4f}")
    
    # 测试形状抑制损失
    suppression_loss = interior_attention.compute_shape_suppression_loss(
        interior_features, attention_mask
    )
    print(f"  形状抑制损失: {suppression_loss.item():.4f}")
    
    # 测试自适应内部注意力
    print(f"\n🎯 测试自适应内部注意力:")
    adaptive_attention = AdaptiveInteriorAttention().to(device)
    adaptive_outputs = adaptive_attention(test_images, test_targets)
    
    print(f"  自适应输出:")
    for key, value in adaptive_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape}")
    
    # 测试自适应损失
    adaptive_losses = adaptive_attention.compute_adaptive_attention_loss(adaptive_outputs, test_targets)
    print(f"  自适应损失:")
    for key, value in adaptive_losses.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.item():.4f}")
        else:
            print(f"    {key}: {value:.4f}")
    
    print(f"\n✅ Interior-Only注意力机制测试完成")

if __name__ == '__main__':
    test_interior_attention()
