# parameter_analysis.py
# 颜色头参数量详细分析

import torch
import torch.nn as nn
import sys
import os
from collections import defaultdict

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from complete_enhanced_color_head import CompleteEnhancedColorHead
except ImportError as e:
    print(f"警告: 无法导入模块: {e}")
    exit(1)

def analyze_parameters():
    """分析颜色头参数量"""
    print("🎨 颜色头参数量详细分析")
    print("="*60)
    
    # 创建颜色头
    color_head = CompleteEnhancedColorHead(input_dim=64, num_classes=43, color_dim=7)
    
    # 总体统计
    total_params = sum(p.numel() for p in color_head.parameters())
    trainable_params = sum(p.numel() for p in color_head.parameters() if p.requires_grad)
    
    print(f"📊 总体统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024:.2f} KB (float32)")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 按模块分组统计
    module_params = defaultdict(int)
    module_details = defaultdict(list)
    
    for name, param in color_head.named_parameters():
        # 提取模块名
        module_name = name.split('.')[0] if '.' in name else name
        module_params[module_name] += param.numel()
        module_details[module_name].append((name, param.numel(), param.shape))
    
    print(f"\n📋 各模块参数量:")
    print("-" * 50)
    
    for module_name in sorted(module_params.keys()):
        params_count = module_params[module_name]
        percentage = params_count / total_params * 100
        print(f"{module_name:25s}: {params_count:8,} 参数 ({percentage:5.1f}%)")
        
        # 显示详细信息
        for param_name, param_count, param_shape in module_details[module_name]:
            print(f"  └─ {param_name:30s}: {param_count:6,} 参数 {param_shape}")
    
    # 按层类型统计
    layer_types = defaultdict(int)
    for name, module in color_head.named_modules():
        if isinstance(module, nn.Linear):
            layer_types['Linear'] += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Conv2d):
            layer_types['Conv2d'] += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.BatchNorm1d):
            layer_types['BatchNorm1d'] += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.BatchNorm2d):
            layer_types['BatchNorm2d'] += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Parameter):
            layer_types['Parameter'] += module.numel()
    
    print(f"\n🔧 按层类型统计:")
    print("-" * 30)
    for layer_type, count in sorted(layer_types.items()):
        percentage = count / total_params * 100
        print(f"{layer_type:15s}: {count:8,} 参数 ({percentage:5.1f}%)")
    
    # 计算FLOPs (简化估算)
    print(f"\n⚡ 计算复杂度估算:")
    print("-" * 30)
    
    # 假设输入batch_size=32, 特征维度=64
    batch_size = 32
    input_dim = 64
    
    # 主要计算路径
    flops = 0
    
    # ShapeDecorrelator
    flops += batch_size * (64 * 32 + 32 * 16) * 2  # 两个extractor
    flops += batch_size * 32 * 16  # decorr_projection
    
    # Color Feature Extractor
    flops += batch_size * (64 * 64 + 64 * 32 + 32 * 16)
    
    # Multimodal Fusion
    flops += batch_size * (128 * 64 + 64 * 32 + 32 * 16)
    
    # Color Semantic Head
    flops += batch_size * (16 * 8 + 8 * 43)
    
    # Color Detector
    flops += batch_size * (16 * 8 + 8 * 7)
    
    print(f"  估算FLOPs: {flops:,}")
    print(f"  每样本FLOPs: {flops // batch_size:,}")
    
    # 内存使用估算
    print(f"\n💾 内存使用估算:")
    print("-" * 30)
    
    # 模型参数内存
    param_memory = total_params * 4  # float32 = 4 bytes
    
    # 激活内存 (简化估算)
    activation_memory = batch_size * (64 + 16 + 16 + 43 + 7) * 4  # 主要激活
    
    # 梯度内存
    gradient_memory = total_params * 4
    
    total_memory = param_memory + activation_memory + gradient_memory
    
    print(f"  参数内存: {param_memory / 1024 / 1024:.2f} MB")
    print(f"  激活内存: {activation_memory / 1024 / 1024:.2f} MB")
    print(f"  梯度内存: {gradient_memory / 1024 / 1024:.2f} MB")
    print(f"  总内存: {total_memory / 1024 / 1024:.2f} MB")
    
    # 与其他模型对比
    print(f"\n📈 参数量对比:")
    print("-" * 30)
    
    comparisons = {
        "ResNet-18": 11_689_512,
        "ResNet-50": 25_557_032,
        "VGG-16": 138_357_544,
        "我们的颜色头": total_params,
        "简单MLP (64→32→43)": 64*32 + 32*43 + 32 + 43,
        "CNN (3x3, 64通道)": 3*3*3*64 + 64,
    }
    
    for model_name, params in comparisons.items():
        print(f"{model_name:20s}: {params:10,} 参数")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'module_params': dict(module_params),
        'layer_types': dict(layer_types)
    }

def compare_with_baseline():
    """与基线模型对比"""
    print(f"\n🔄 与基线模型对比:")
    print("="*40)
    
    # 基线模型 - 简单的颜色头
    class BaselineColorHead(nn.Module):
        def __init__(self, input_dim=64, num_classes=43):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            return self.classifier(x)
    
    baseline = BaselineColorHead()
    baseline_params = sum(p.numel() for p in baseline.parameters())
    
    # 我们的增强颜色头
    enhanced = CompleteEnhancedColorHead(input_dim=64, num_classes=43)
    enhanced_params = sum(p.numel() for p in enhanced.parameters())
    
    print(f"基线颜色头参数量: {baseline_params:,}")
    print(f"增强颜色头参数量: {enhanced_params:,}")
    print(f"参数量增加: {enhanced_params / baseline_params:.1f}x")
    print(f"参数量增加: {enhanced_params - baseline_params:,} 参数")
    
    # 功能对比
    print(f"\n🎯 功能对比:")
    print(f"基线颜色头:")
    print(f"  - 简单线性分类")
    print(f"  - 无归纳偏置")
    print(f"  - 无形状去相关")
    print(f"  - 无符号知识集成")
    
    print(f"\n增强颜色头:")
    print(f"  - 多模态特征融合")
    print(f"  - 强归纳偏置")
    print(f"  - 形状去相关机制")
    print(f"  - 符号知识驱动")
    print(f"  - 自适应权重学习")
    print(f"  - 6种损失函数")

if __name__ == '__main__':
    results = analyze_parameters()
    compare_with_baseline()
    
    print(f"\n✅ 参数量分析完成!")
    print(f"我们的颜色头是一个轻量级但功能强大的模块，")
    print(f"在保持较低参数量的同时实现了强归纳偏置。")
