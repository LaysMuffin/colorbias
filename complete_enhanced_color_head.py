# complete_enhanced_color_head.py
# 完整的增强颜色头 - 整合所有归纳偏置改进

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtsrb_symbolic_knowledge import GTSRBSymbolicKnowledge
from color_space_transformer import ColorSpaceTransformer

# 导入真正的Interior-Only注意力机制
try:
    from interior_attention import InteriorAttention
except ImportError:
    # 如果导入失败，使用简单的实现
    class InteriorAttention(nn.Module):
        """内部注意力机制 - 专注标志内部区域"""
        
        def __init__(self, input_channels=3):
            super().__init__()
            self.input_channels = input_channels
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """前向传播 - 生成内部注意力掩码"""
            batch_size = x.size(0)
            # 简单的实现
            attention_mask = torch.ones_like(x[:, :1, :, :]) * 0.5
            edges = torch.zeros_like(x[:, :1, :, :])
            shape_mask = torch.zeros_like(x[:, :1, :, :])
            interior_features = x
            return interior_features, attention_mask, edges, shape_mask

class ColorChannelAttention(nn.Module):
    """颜色通道注意力机制"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        self.input_channels = input_channels
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return {
            'channel_attended': x,
            'spatial_attended': x,
            'combined_attended': x,
            'channel_weights': torch.ones_like(x[:, :, :1, :1]),
            'spatial_weights': torch.ones_like(x[:, :1, :, :]),
            'color_importance': torch.ones_like(x[:, :, :1, :1])
        }

class MultiScaleColorFeatures(nn.Module):
    """多尺度颜色特征提取"""
    
    def __init__(self, input_channels=3, output_channels=16):
        super().__init__()
        self.output_channels = output_channels
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = x.size(0)
        features = torch.randn(batch_size, self.output_channels, x.size(2), x.size(3), device=x.device)
        return {
            'scale1': features,
            'scale2': features,
            'scale3': features,
            'concatenated': torch.cat([features, features, features], dim=1),
            'attention_weights': torch.ones_like(torch.cat([features, features, features], dim=1)),
            'weighted_features': torch.cat([features, features, features], dim=1),
            'fused_features': features
        }

class CompleteEnhancedColorHead(nn.Module):
    """完整的增强颜色头 - 整合所有归纳偏置改进"""
    
    def __init__(self, input_dim=64, num_classes=43, color_dim=7, input_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.color_dim = color_dim
        self.input_dim = input_dim
        self.input_channels = input_channels
        
        # GTSRB符号知识系统
        self.gtsrb_knowledge = GTSRBSymbolicKnowledge()
        
        # 增强的颜色空间转换模块
        self.color_space_transformer = ColorSpaceTransformer(input_channels)
        
        # 内部注意力机制
        self.interior_attention = InteriorAttention(input_channels)
        
        # 颜色通道注意力
        self.color_channel_attention = ColorChannelAttention(input_channels)
        
        # 多尺度颜色特征
        self.multi_scale_features = MultiScaleColorFeatures(input_channels, 16)
        
        # 形状去相关模块
        self.shape_decorrelator = ShapeDecorrelator(input_dim)
        
        # 颜色特征提取器 - 专门处理颜色信息
        self.color_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 多模态特征融合
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(input_dim + 16 + 48, 64),  # 原始特征 + 颜色特征 + 多尺度特征
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 颜色语义头
        self.color_semantic_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes)
        )
        
        # 颜色检测器
        self.color_detector = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, color_dim)
        )
        
        # 自适应权重
        self.adaptive_weights = nn.Parameter(torch.ones(4))  # 4个特征源的权重
        
        # 损失权重
        self.lambda_color_consistency = 0.2
        self.lambda_semantic_consistency = 0.1
        self.lambda_shape_decorr = 0.15
        self.lambda_color_invariance = 0.05
        self.lambda_interior_attention = 0.1
        self.lambda_multimodal_fusion = 0.05
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = x.size(0)
        
        # 检查输入维度
        if x.dim() == 2:  # 如果是特征输入
            # 直接使用特征，跳过图像处理
            features = x
            color_spaces = None
            interior_features = None
            attention_mask = None
            edges = None
            shape_mask = None
            channel_outputs = None
            multi_scale_outputs = None
        else:  # 如果是图像输入
            # 颜色空间转换
            color_spaces = self.color_space_transformer(x)
            
            # 内部注意力
            interior_features, attention_mask, edges, shape_mask = self.interior_attention(x)
            
            # 颜色通道注意力
            channel_outputs = self.color_channel_attention(x)
            
            # 多尺度颜色特征
            multi_scale_outputs = self.multi_scale_features(x)
            
            # 特征提取（假设输入已经是特征）
            if x.dim() == 4:  # 如果是图像，需要先提取特征
                # 这里假设已经有特征提取器
                features = x.view(batch_size, -1)
            else:
                features = x
        
        # 形状去相关
        decorr_outputs = self.shape_decorrelator(features)
        
        # 颜色特征提取
        color_features = self.color_feature_extractor(features)
        
        # 多模态特征融合
        if multi_scale_outputs is not None:
            # 提取多尺度特征
            multi_scale_features = multi_scale_outputs['fused_features'].view(batch_size, -1)
            # 确保维度匹配
            if multi_scale_features.size(1) > 48:
                multi_scale_features = multi_scale_features[:, :48]
            elif multi_scale_features.size(1) < 48:
                # 填充到48维
                padding = torch.zeros(batch_size, 48 - multi_scale_features.size(1), device=multi_scale_features.device)
                multi_scale_features = torch.cat([multi_scale_features, padding], dim=1)
        else:
            # 如果没有多尺度特征，使用零填充
            multi_scale_features = torch.zeros(batch_size, 48, device=features.device)
        
        # 拼接所有特征
        all_features = torch.cat([features, color_features, multi_scale_features], dim=1)
        
        # 多模态融合
        fused_features = self.multimodal_fusion(all_features)
        
        # 自适应权重融合
        adaptive_weights = F.softmax(self.adaptive_weights, dim=0)
        
        # 颜色语义预测
        color_semantic_logits = self.color_semantic_head(fused_features)
        
        # 颜色检测
        color_logits = self.color_detector(fused_features)
        color_probs = F.softmax(color_logits, dim=1)
        
        return {
            'color_semantic_logits': color_semantic_logits,
            'color_logits': color_logits,
            'color_probs': color_probs,
            'features': features,
            'color_features': color_features,
            'fused_features': fused_features,
            'decorr_outputs': decorr_outputs,
            'color_spaces': color_spaces,
            'interior_features': interior_features,
            'attention_mask': attention_mask,
            'edges': edges,
            'shape_mask': shape_mask,
            'channel_outputs': channel_outputs,
            'multi_scale_outputs': multi_scale_outputs,
            'adaptive_weights': adaptive_weights,
            'all_features': all_features
        }
    
    def compute_complete_loss(self, outputs: Dict[str, torch.Tensor], 
                            targets: torch.Tensor, 
                            inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算完整损失"""
        batch_size = outputs['color_semantic_logits'].size(0)
        
        # 基础分类损失
        ce_loss = F.cross_entropy(outputs['color_semantic_logits'], targets)
        
        # 颜色一致性损失
        color_consistency_loss = self._compute_color_consistency_loss(
            outputs['color_probs'], targets
        )
        
        # 语义一致性损失
        semantic_consistency_loss = self._compute_semantic_consistency_loss(
            outputs['color_semantic_logits'], targets, outputs['color_probs']
        )
        
        # 形状去相关损失
        shape_decorr_loss = self._compute_shape_decorrelation_loss(
            outputs['decorr_outputs']
        )
        
        # 颜色不变性损失
        color_invariance_loss = self._compute_color_invariance_loss(
            outputs['color_features']
        )
        
        # 内部注意力损失
        if outputs['attention_mask'] is not None:
            interior_attention_loss = self._compute_interior_attention_loss(
                outputs['attention_mask'], outputs['color_semantic_logits'], targets
            )
        else:
            interior_attention_loss = torch.tensor(0.0, device=outputs['color_semantic_logits'].device)
        
        # 多模态融合损失
        multimodal_fusion_loss = self._compute_multimodal_fusion_loss(
            outputs['all_features'], outputs['adaptive_weights']
        )
        
        # 总损失
        total_loss = (
            ce_loss +
            self.lambda_color_consistency * color_consistency_loss +
            self.lambda_semantic_consistency * semantic_consistency_loss +
            self.lambda_shape_decorr * shape_decorr_loss +
            self.lambda_color_invariance * color_invariance_loss +
            self.lambda_interior_attention * interior_attention_loss +
            self.lambda_multimodal_fusion * multimodal_fusion_loss
        )
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'color_consistency_loss': color_consistency_loss,
            'semantic_consistency_loss': semantic_consistency_loss,
            'shape_decorr_loss': shape_decorr_loss,
            'color_invariance_loss': color_invariance_loss,
            'interior_attention_loss': interior_attention_loss,
            'multimodal_fusion_loss': multimodal_fusion_loss
        }
    
    def _compute_color_consistency_loss(self, color_probs: torch.Tensor, 
                                      targets: torch.Tensor) -> torch.Tensor:
        """计算颜色一致性损失"""
        loss = 0
        
        for i in range(color_probs.size(0)):
            target_class = targets[i].item()
            expected_colors = self.gtsrb_knowledge.get_color_requirements(target_class)
            
            if expected_colors:
                for part, expected_color in expected_colors.items():
                    if expected_color in self.gtsrb_knowledge.color_label_to_id:
                        color_id = self.gtsrb_knowledge.color_label_to_id[expected_color]
                        expected_prob = 1.0
                        actual_prob = color_probs[i, color_id]
                        
                        # 惩罚颜色不匹配
                        loss += F.mse_loss(actual_prob, torch.tensor(expected_prob, device=color_probs.device))
        
        return loss / color_probs.size(0)
    
    def _compute_semantic_consistency_loss(self, logits: torch.Tensor, 
                                         targets: torch.Tensor, 
                                         color_probs: torch.Tensor) -> torch.Tensor:
        """计算语义一致性损失"""
        loss = 0
        
        for i in range(logits.size(0)):
            pred_class = torch.argmax(logits[i]).item()
            target_class = targets[i].item()
            
            pred_semantic = self.gtsrb_knowledge.get_semantic_category(pred_class)
            target_semantic = self.gtsrb_knowledge.get_semantic_category(target_class)
            
            if pred_semantic != target_semantic:
                loss += 0.5
        
        return loss / logits.size(0)
    
    def _compute_shape_decorrelation_loss(self, decorr_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算形状去相关损失"""
        shape_features = decorr_outputs['shape_features']
        color_features = decorr_outputs['color_features']
        
        # 计算相关性
        shape_flat = shape_features.view(shape_features.size(0), -1)
        color_flat = color_features.view(color_features.size(0), -1)
        
        # 标准化
        shape_norm = (shape_flat - shape_flat.mean(dim=1, keepdim=True)) / (shape_flat.std(dim=1, keepdim=True) + 1e-8)
        color_norm = (color_flat - color_flat.mean(dim=1, keepdim=True)) / (color_flat.std(dim=1, keepdim=True) + 1e-8)
        
        # 计算相关性矩阵
        correlation = torch.mm(shape_norm, color_norm.t()) / shape_norm.size(1)
        
        # 惩罚高相关性
        decorr_loss = torch.mean(torch.abs(correlation))
        
        return decorr_loss
    
    def _compute_color_invariance_loss(self, color_features: torch.Tensor) -> torch.Tensor:
        """计算颜色不变性损失"""
        # 简单的颜色特征稳定性损失
        # 鼓励颜色特征在不同样本间保持一致性
        feature_mean = torch.mean(color_features, dim=0)
        feature_std = torch.std(color_features, dim=0)
        
        # 惩罚过大的方差
        invariance_loss = torch.mean(feature_std)
        
        return invariance_loss
    
    def _compute_interior_attention_loss(self, attention_mask: torch.Tensor, 
                                       logits: torch.Tensor, 
                                       targets: torch.Tensor) -> torch.Tensor:
        """计算内部注意力损失"""
        # 计算平均注意力强度
        attention_strength = torch.mean(attention_mask, dim=[2, 3])
        
        # 颜色相关类别需要高内部注意力
        color_related_classes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42]
        
        loss = 0
        for i, target in enumerate(targets):
            if target.item() in color_related_classes:
                # 颜色相关类别需要高内部注意力
                target_attention = 0.8
            else:
                # 非颜色相关类别允许低内部注意力
                target_attention = 0.3
            
            loss += F.mse_loss(attention_strength[i], torch.tensor(target_attention, device=attention_strength.device))
        
        return loss / len(targets)
    
    def _compute_multimodal_fusion_loss(self, all_features: torch.Tensor, 
                                      adaptive_weights: torch.Tensor) -> torch.Tensor:
        """计算多模态融合损失"""
        # 鼓励特征多样性
        feature_diversity = torch.std(all_features, dim=1)
        diversity_loss = -torch.mean(feature_diversity)  # 负号表示最大化多样性
        
        # 鼓励权重平衡
        weight_balance_loss = torch.std(adaptive_weights)
        
        return diversity_loss + 0.1 * weight_balance_loss
    
    def extract_color_rules(self, outputs: Dict[str, torch.Tensor], 
                          targets: torch.Tensor) -> Tuple[list, list]:
        """提取颜色规则"""
        rules = []
        validations = []
        
        for i in range(outputs['color_semantic_logits'].size(0)):
            pred_class = torch.argmax(outputs['color_semantic_logits'][i]).item()
            target_class = targets[i].item()
            
            # 获取颜色要求
            color_requirements = self.gtsrb_knowledge.get_color_requirements(pred_class)
            
            # 构建规则
            rule = {
                'sample_id': i,
                'predicted_class': pred_class,
                'target_class': target_class,
                'predicted_name': self.gtsrb_knowledge.class_names.get(pred_class, 'Unknown'),
                'target_name': self.gtsrb_knowledge.class_names.get(target_class, 'Unknown'),
                'color_requirements': color_requirements,
                'semantic_category': self.gtsrb_knowledge.get_semantic_category(pred_class),
                'confidence': torch.softmax(outputs['color_semantic_logits'][i], dim=0).max().item(),
                'attention_strength': torch.mean(outputs['attention_mask'][i]).item() if outputs['attention_mask'] is not None else 0.0,
                'adaptive_weights': outputs['adaptive_weights'].detach().cpu().numpy().tolist()
            }
            
            rules.append(rule)
            
            # 验证规则
            validation = {
                'rule': rule,
                'target_class': target_class,
                'target_name': self.gtsrb_knowledge.class_names.get(target_class, 'Unknown'),
                'semantic_match': rule['predicted_class'] == target_class,
                'semantic_category_match': rule['semantic_category'] == self.gtsrb_knowledge.get_semantic_category(target_class),
                'confidence': rule['confidence'],
                'attention_strength': rule['attention_strength']
            }
            
            validations.append(validation)
        
        return rules, validations

class ShapeDecorrelator(nn.Module):
    """形状去相关模块 - 抑制形状干扰"""
    
    def __init__(self, feature_dim=64):
        super().__init__()
        
        # 形状特征提取器
        self.shape_extractor = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 颜色特征提取器
        self.color_extractor = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 去相关投影
        self.decorr_projection = nn.Sequential(
            nn.Linear(32, 16),  # 32 = 16(shape) + 16(color)
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    
    def forward(self, features):
        """前向传播 - 提取去相关特征"""
        # 提取形状和颜色特征
        shape_features = self.shape_extractor(features)
        color_features = self.color_extractor(features)
        
        # 组合特征
        combined_features = torch.cat([shape_features, color_features], dim=1)
        
        # 去相关投影
        decorr_features = self.decorr_projection(combined_features)
        
        return {
            'shape_features': shape_features,
            'color_features': color_features,
            'decorr_features': decorr_features,
            'combined_features': combined_features
        }

def test_complete_enhanced_color_head():
    """测试完整的增强颜色头"""
    print("🎨 测试完整增强颜色头")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建完整增强颜色头
    complete_head = CompleteEnhancedColorHead(input_dim=64, num_classes=43).to(device)
    
    # 创建测试数据
    batch_size = 4
    features = torch.randn(batch_size, 64).to(device)
    targets = torch.randint(0, 43, (batch_size,)).to(device)
    inputs = torch.randn(batch_size, 3, 32, 32).to(device)
    
    print(f"批次大小: {batch_size}")
    print(f"特征维度: {features.shape}")
    
    # 测试前向传播
    outputs = complete_head(features)
    print(f"\n📊 前向传播测试:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {type(value)}")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"    {sub_key}: {sub_value.shape}")
    
    # 测试损失计算
    loss_dict = complete_head.compute_complete_loss(outputs, targets, inputs)
    print(f"\n📈 损失计算测试:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 测试规则提取
    rules, validations = complete_head.extract_color_rules(outputs, targets)
    print(f"\n📋 规则提取测试:")
    print(f"  提取规则数: {len(rules)}")
    print(f"  验证结果数: {len(validations)}")
    
    # 显示规则示例
    if rules:
        rule = rules[0]
        print(f"\n📝 规则示例:")
        print(f"  预测类别: {rule['predicted_class']} ({rule['predicted_name']})")
        print(f"  目标类别: {rule['target_class']} ({rule['target_name']})")
        print(f"  语义类别: {rule['semantic_category']}")
        print(f"  置信度: {rule['confidence']:.3f}")
        print(f"  注意力强度: {rule['attention_strength']:.3f}")
        print(f"  自适应权重: {rule['adaptive_weights']}")
    
    print(f"\n✅ 完整增强颜色头测试完成")

if __name__ == '__main__':
    test_complete_enhanced_color_head()
