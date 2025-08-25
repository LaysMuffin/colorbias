# train_with_gtsrb.py
# 使用真实GTSRB数据集训练增强颜色头

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from datasets.gtsrb_dataset import get_gtsrb_dataloaders
    from stable_training_model import StableTrainingModel
    from gtsrb_symbolic_knowledge import GTSRBSymbolicKnowledge
except ImportError as e:
    print(f"警告: 无法导入某些模块: {e}")
    print("将使用模拟数据")

from complete_enhanced_color_head import CompleteEnhancedColorHead as EnhancedColorSemanticHead
from color_augmentation import AdvancedColorSpecificAugmentation as ColorSpecificAugmentation, ColorRobustnessLoss
from training_strategies import StagedTraining, CurriculumLearning

class GTSRBEnhancedColorModel(nn.Module):
    """GTSRB增强颜色模型 - 整合所有改进"""
    
    def __init__(self, base_model=None):
        super().__init__()
        
        # 如果没有提供基础模型，创建一个简单的
        if base_model is None:
            self.base_model = SimpleBaseModel()
        else:
            self.base_model = base_model
        
        # 获取特征维度
        if hasattr(self.base_model, 'prototype_features'):
            feature_dim = self.base_model.prototype_features.size(1)
        else:
            feature_dim = 1161  # 默认维度
        
        # 增强颜色头
        self.enhanced_color_head = EnhancedColorSemanticHead(
            input_dim=feature_dim, 
            num_classes=43
        )
        
        # 颜色增强
        self.color_augmentation = ColorSpecificAugmentation()
        self.color_robustness_loss = ColorRobustnessLoss()
        
        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # 符号知识
        self.gtsrb_knowledge = GTSRBSymbolicKnowledge()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 基础模型前向传播
        base_outputs = self.base_model(x)
        
        # 获取特征
        if 'features' in base_outputs:
            features = base_outputs['features']
        elif 'prototype_features' in base_outputs:
            features = base_outputs['prototype_features']
        else:
            # 如果没有特征，使用输入
            features = x.view(x.size(0), -1)
        
        # 增强颜色头前向传播
        color_outputs = self.enhanced_color_head(features)
        
        # 融合预测
        if 'logits' in base_outputs:
            base_logits = base_outputs['logits']
        else:
            base_logits = torch.zeros(x.size(0), 43, device=x.device)
        
        color_logits = color_outputs['color_semantic_logits']
        
        # 自适应融合
        fusion_weight = torch.sigmoid(self.fusion_weight)
        final_logits = fusion_weight * color_logits + (1 - fusion_weight) * base_logits
        
        return {
            'base_logits': base_logits,
            'color_logits': color_logits,
            'final_logits': final_logits,
            'color_outputs': color_outputs,
            'base_outputs': base_outputs,
            'fusion_weight': fusion_weight,
            'features': features
        }
    
    def compute_enhanced_loss(self, outputs: Dict[str, torch.Tensor], 
                            targets: torch.Tensor, 
                            inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算增强损失"""
        # 基础分类损失
        ce_loss = nn.CrossEntropyLoss()(outputs['final_logits'], targets)
        
        # 颜色头损失
        color_loss_dict = self.enhanced_color_head.compute_complete_loss(
            outputs['color_outputs'], targets, inputs
        )
        
        # 融合损失
        fusion_loss = nn.MSELoss()(outputs['final_logits'], outputs['base_logits'])
        
        # 颜色鲁棒性损失
        if hasattr(self, 'color_robustness_loss'):
            robustness_loss = self.color_robustness_loss(
                outputs['features'], outputs['features'], 
                outputs['final_logits'], outputs['final_logits']
            )
            robustness_total = robustness_loss['total_loss']
        else:
            robustness_total = torch.tensor(0.0, device=outputs['final_logits'].device)
        
        # 总损失
        total_loss = (
            ce_loss + 
            color_loss_dict['total_loss'] * 0.5 +
            fusion_loss * 0.1 +
            robustness_total * 0.05
        )
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'color_loss': color_loss_dict['total_loss'],
            'fusion_loss': fusion_loss,
            'robustness_loss': robustness_total
        }

class SimpleBaseModel(nn.Module):
    """简单的基础模型 - 用于测试"""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(3072, 512),  # 32x32x3 = 3072
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, 43)
        self.prototype_features = torch.randn(43, 128)  # 模拟原型特征
    
    def forward(self, x):
        # 展平输入
        x_flat = x.view(x.size(0), -1)
        features = self.features(x_flat)
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'prototype_features': features
        }

def create_mock_dataloaders():
    """创建模拟数据加载器"""
    print("📊 创建模拟GTSRB数据加载器")
    
    class MockGTSRBDataset:
        def __init__(self, size=1000, train=True):
            self.size = size
            self.train = train
            # 创建模拟图像数据
            self.data = torch.randn(size, 3, 32, 32)
            self.targets = torch.randint(0, 43, (size,))
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
        
        def __len__(self):
            return self.size
    
    train_dataset = MockGTSRBDataset(size=2000, train=True)
    test_dataset = MockGTSRBDataset(size=500, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_enhanced_model_with_gtsrb(
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    use_real_data: bool = False,
    model_name: str = "gtsrb_enhanced_color_model"
):
    """使用GTSRB数据集训练增强颜色模型"""
    
    print("🎨 GTSRB增强颜色模型训练")
    print("="*60)
    print(f"训练配置:")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  lr: {lr}")
    print(f"  use_real_data: {use_real_data}")
    print(f"  model_name: {model_name}")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 数据加载器
    if use_real_data:
        try:
            print("\n📊 加载真实GTSRB数据集...")
            train_loader, test_loader = get_gtsrb_dataloaders(
                data_root='./data/GTSRB',
                batch_size=batch_size,
                num_workers=4,
                use_augmentation=True
            )
            print("✅ 真实GTSRB数据集加载成功")
        except Exception as e:
            print(f"❌ 真实数据集加载失败: {e}")
            print("📊 使用模拟数据集...")
            train_loader, test_loader = create_mock_dataloaders()
    else:
        print("📊 使用模拟数据集...")
        train_loader, test_loader = create_mock_dataloaders()
    
    # 创建模型
    print("\n🏗️ 创建增强颜色模型...")
    enhanced_model = GTSRBEnhancedColorModel().to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in enhanced_model.parameters())
    trainable_params = sum(p.numel() for p in enhanced_model.parameters() if p.requires_grad)
    print(f"模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 优化器和调度器
    optimizer = optim.AdamW(enhanced_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # 训练历史
    train_history = {
        'loss': [], 'accuracy': [], 'color_accuracy': [], 'fusion_weight': []
    }
    val_history = {
        'loss': [], 'accuracy': [], 'color_accuracy': []
    }
    
    # 训练循环
    print(f"\n🚀 开始训练...")
    start_time = time.time()
    best_val_acc = 0
    
    for epoch in range(epochs):
        # 训练阶段
        enhanced_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        color_correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            outputs = enhanced_model(data)
            loss_dict = enhanced_model.compute_enhanced_loss(outputs, target, data)
            
            # 反向传播
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(enhanced_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            train_loss += loss_dict['total_loss'].item()
            
            # 计算准确率
            _, predicted = torch.max(outputs['final_logits'], 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # 颜色头准确率
            _, color_predicted = torch.max(outputs['color_logits'], 1)
            color_correct += (color_predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss_dict['total_loss'].item():.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 验证阶段
        enhanced_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_color_correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = enhanced_model(data)
                loss_dict = enhanced_model.compute_enhanced_loss(outputs, target, data)
                
                val_loss += loss_dict['total_loss'].item()
                
                _, predicted = torch.max(outputs['final_logits'], 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                _, color_predicted = torch.max(outputs['color_logits'], 1)
                val_color_correct += (color_predicted == target).sum().item()
        
        # 计算准确率
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_color_acc = 100. * color_correct / train_total
        val_color_acc = 100. * val_color_correct / val_total
        
        # 记录历史
        train_history['loss'].append(train_loss / len(train_loader))
        train_history['accuracy'].append(train_acc)
        train_history['color_accuracy'].append(train_color_acc)
        train_history['fusion_weight'].append(outputs['fusion_weight'].item())
        
        val_history['loss'].append(val_loss / len(test_loader))
        val_history['accuracy'].append(val_acc)
        val_history['color_accuracy'].append(val_color_acc)
        
        # 打印结果
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%, Color Acc: {train_color_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss/len(test_loader):.4f}, Acc: {val_acc:.2f}%, Color Acc: {val_color_acc:.2f}%")
        print(f"  Fusion Weight: {outputs['fusion_weight'].item():.3f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': enhanced_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_history': train_history,
                'val_history': val_history
            }, f'checkpoints/{model_name}_best.pth')
            print(f"  💾 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
    
    # 训练完成
    total_time = time.time() - start_time
    print(f"\n🎉 训练完成!")
    print(f"总训练时间: {total_time/60:.2f}分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    return enhanced_model, train_history, val_history

def evaluate_enhanced_model(model: GTSRBEnhancedColorModel, test_loader: DataLoader):
    """评估增强模型"""
    print("\n📊 评估增强颜色模型")
    print("="*40)
    
    device = next(model.parameters()).device
    model.eval()
    
    total_correct = 0
    total_samples = 0
    color_correct = 0
    base_correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            # 最终预测
            _, predicted = torch.max(outputs['final_logits'], 1)
            total_correct += (predicted == target).sum().item()
            
            # 颜色头预测
            _, color_predicted = torch.max(outputs['color_logits'], 1)
            color_correct += (color_predicted == target).sum().item()
            
            # 基础模型预测
            _, base_predicted = torch.max(outputs['base_logits'], 1)
            base_correct += (base_predicted == target).sum().item()
            
            total_samples += target.size(0)
    
    final_acc = 100. * total_correct / total_samples
    color_acc = 100. * color_correct / total_samples
    base_acc = 100. * base_correct / total_samples
    
    print(f"最终准确率: {final_acc:.2f}%")
    print(f"颜色头准确率: {color_acc:.2f}%")
    print(f"基础模型准确率: {base_acc:.2f}%")
    print(f"融合权重: {outputs['fusion_weight'].item():.3f}")
    
    return {
        'final_accuracy': final_acc,
        'color_accuracy': color_acc,
        'base_accuracy': base_acc,
        'fusion_weight': outputs['fusion_weight'].item()
    }

def main():
    """主函数"""
    print("🎨 GTSRB增强颜色模型训练脚本")
    print("="*60)
    
    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练配置
    config = {
        'epochs': 15,
        'batch_size': 32,
        'lr': 0.001,
        'use_real_data': False,  # 设置为True使用真实GTSRB数据集
        'model_name': 'gtsrb_enhanced_color_model'
    }
    
    # 训练模型
    enhanced_model, train_history, val_history = train_enhanced_model_with_gtsrb(**config)
    
    # 创建测试数据加载器进行评估
    _, test_loader = create_mock_dataloaders()
    
    # 评估模型
    evaluation_results = evaluate_enhanced_model(enhanced_model, test_loader)
    
    print(f"\n📈 训练总结:")
    print(f"最佳验证准确率: {max(val_history['accuracy']):.2f}%")
    print(f"最终测试准确率: {evaluation_results['final_accuracy']:.2f}%")
    print(f"颜色头贡献: {evaluation_results['color_accuracy']:.2f}%")
    print(f"融合权重: {evaluation_results['fusion_weight']:.3f}")

if __name__ == '__main__':
    main()
