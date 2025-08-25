# test_all_modules.py
# 测试所有颜色头和偏置模块

import torch
import torch.nn as nn
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_all_modules():
    """测试所有模块"""
    print("🎯 测试所有颜色头和偏置模块")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 4
    features = torch.randn(batch_size, 64).to(device)
    targets = torch.randint(0, 43, (batch_size,)).to(device)
    images = torch.randn(batch_size, 3, 32, 32).to(device)
    
    print(f"\n📊 测试数据:")
    print(f"  特征维度: {features.shape}")
    print(f"  目标维度: {targets.shape}")
    print(f"  图像维度: {images.shape}")
    
    # 测试1: 颜色空间转换模块
    print(f"\n🎨 测试1: 颜色空间转换模块")
    try:
        from color_space_transformer import ColorSpaceTransformer
        color_transformer = ColorSpaceTransformer().to(device)
        color_outputs = color_transformer(images)
        print(f"  ✅ 颜色空间转换模块测试通过")
        print(f"    输出形状: {color_outputs['fused'].shape}")
    except Exception as e:
        print(f"  ❌ 颜色空间转换模块测试失败: {e}")
    
    # 测试2: 增强颜色头
    print(f"\n🎨 测试2: 增强颜色头")
    try:
        from complete_enhanced_color_head import CompleteEnhancedColorHead as EnhancedColorSemanticHead
        enhanced_head = EnhancedColorSemanticHead(input_dim=64, num_classes=43).to(device)
        outputs = enhanced_head(features)
        loss_dict = enhanced_head.compute_complete_loss(outputs, targets, images)
        print(f"  ✅ 增强颜色头测试通过")
        print(f"    总损失: {loss_dict['total_loss'].item():.4f}")
    except Exception as e:
        print(f"  ❌ 增强颜色头测试失败: {e}")
    
    # 测试3: 颜色增强模块
    print(f"\n🎨 测试3: 颜色增强模块")
    try:
        from color_augmentation import AdvancedColorSpecificAugmentation, ColorRobustnessLoss
        color_aug = AdvancedColorSpecificAugmentation()
        color_robustness = ColorRobustnessLoss()
        
        augmented_images = color_aug(images, targets)
        # 确保数据类型正确
        predictions_orig = torch.randn(batch_size, 43).float().to(device)
        predictions_aug = torch.randn(batch_size, 43).float().to(device)
        robustness_loss = color_robustness(features, features, predictions_orig, predictions_aug)
        print(f"  ✅ 颜色增强模块测试通过")
        print(f"    增强图像形状: {augmented_images.shape}")
        print(f"    鲁棒性损失: {robustness_loss['total_loss'].item():.4f}")
    except Exception as e:
        print(f"  ❌ 颜色增强模块测试失败: {e}")
    
    # 测试4: 训练策略模块
    print(f"\n🎨 测试4: 训练策略模块")
    try:
        from training_strategies import StagedTraining, CurriculumLearning
        
        # 创建简单的测试模型
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 43)
                )
            
            def forward(self, x):
                return {'logits': self.features(x)}
        
        # 创建简单的数据加载器
        class TestDataset:
            def __init__(self, size=100):
                self.size = size
                self.data = torch.randn(size, 64)
                self.targets = torch.randint(0, 43, (size,))
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
            
            def __len__(self):
                return self.size
        
        from torch.utils.data import DataLoader
        dataset = TestDataset()
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        model = TestModel().to(device)
        staged_trainer = StagedTraining(model, train_loader, val_loader, device)
        curriculum_trainer = CurriculumLearning(model, train_loader, val_loader, device)
        
        print(f"  ✅ 训练策略模块测试通过")
        print(f"    分阶段训练器: {type(staged_trainer)}")
        print(f"    课程学习器: {type(curriculum_trainer)}")
    except Exception as e:
        print(f"  ❌ 训练策略模块测试失败: {e}")
    
    # 测试5: 完整增强颜色头
    print(f"\n🎨 测试5: 完整增强颜色头")
    try:
        from complete_enhanced_color_head import CompleteEnhancedColorHead
        complete_head = CompleteEnhancedColorHead(input_dim=64, num_classes=43).to(device)
        outputs = complete_head(features)
        loss_dict = complete_head.compute_complete_loss(outputs, targets, images)
        rules, validations = complete_head.extract_color_rules(outputs, targets)
        
        print(f"  ✅ 完整增强颜色头测试通过")
        print(f"    总损失: {loss_dict['total_loss'].item():.4f}")
        print(f"    提取规则数: {len(rules)}")
    except Exception as e:
        print(f"  ❌ 完整增强颜色头测试失败: {e}")
    
    # 测试6: 训练脚本（简化版）
    print(f"\n🎨 测试6: 训练脚本（简化版）")
    try:
        from train_with_gtsrb import GTSRBEnhancedColorModel
        
        # 创建简单的基础模型
        class SimpleBaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Linear(64, 64),  # 保持64维特征
                    nn.ReLU(),
                    nn.Linear(64, 64)  # 输出64维特征
                )
                self.classifier = nn.Linear(64, 43)  # 分类器
                self.prototype_features = torch.randn(43, 64)  # 64维原型特征
            
            def forward(self, x):
                # 确保输入维度正确
                if x.dim() == 2:
                    features = self.feature_extractor(x)
                else:
                    # 如果是图像，先展平
                    x_flat = x.view(x.size(0), -1)
                    features = self.feature_extractor(x_flat)
                
                logits = self.classifier(features)
                return {'logits': logits, 'features': features, 'prototype_features': features}
        
        base_model = SimpleBaseModel().to(device)
        enhanced_model = GTSRBEnhancedColorModel(base_model).to(device)
        
        # 测试前向传播
        outputs = enhanced_model(features)
        loss_dict = enhanced_model.compute_enhanced_loss(outputs, targets, images)
        
        print(f"  ✅ 训练脚本测试通过")
        print(f"    增强模型输出形状: {outputs['final_logits'].shape}")
        print(f"    总损失: {loss_dict['total_loss'].item():.4f}")
    except Exception as e:
        print(f"  ❌ 训练脚本测试失败: {e}")
    
    print(f"\n🎉 所有模块测试完成!")
    print("="*80)

if __name__ == '__main__':
    test_all_modules()
