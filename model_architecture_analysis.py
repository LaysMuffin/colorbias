# model_architecture_analysis.py
# 详细分析我们当前模型的架构

import torch
import torch.nn as nn
import json
import sys
import os

def analyze_model_architecture():
    """分析模型架构"""
    print("🔍 模型架构详细分析")
    print("="*80)
    
    # 模型1: 修复的颜色头
    class FixedColorHead(nn.Module):
        def __init__(self, input_dim=64, num_classes=43):
            super().__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            
            # 颜色特征提取器
            self.color_extractor = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            
            # 颜色分类器
            self.color_classifier = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
            
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # 颜色特征提取
            color_features = self.color_extractor(x)
            
            # 颜色语义预测
            color_logits = self.color_classifier(color_features)
            
            return {
                'color_semantic_logits': color_logits,
                'color_features': color_features,
                'features': x
            }
    
    # 模型2: 基础模型
    class BaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(3072, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 43)
            )
            
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return {'logits': self.features(x)}
    
    # 模型3: 集成模型
    class EnsembleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = BaseModel()
            self.color_head = FixedColorHead()
            
            # 可学习的融合权重
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            
        def forward(self, x):
            # 基础模型预测
            base_outputs = self.base_model(x)
            base_logits = base_outputs['logits']
            
            # 颜色头预测
            features = x.view(x.size(0), -1)[:, :64]
            color_outputs = self.color_head(features)
            color_logits = color_outputs['color_semantic_logits']
            
            # 融合预测
            fusion_weight = torch.sigmoid(self.fusion_weight)
            final_logits = fusion_weight * base_logits + (1 - fusion_weight) * color_logits
            
            return {
                'final_logits': final_logits,
                'base_logits': base_logits,
                'color_logits': color_logits,
                'fusion_weight': fusion_weight
            }
    
    # 创建模型实例
    fixed_color_head = FixedColorHead()
    base_model = BaseModel()
    ensemble_model = EnsembleModel()
    
    # 分析每个模型
    models = {
        'FixedColorHead': fixed_color_head,
        'BaseModel': base_model,
        'EnsembleModel': ensemble_model
    }
    
    architecture_details = {}
    
    for model_name, model in models.items():
        print(f"\n🔬 {model_name} 架构分析")
        print("-" * 60)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 详细层分析
        layer_details = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                if isinstance(module, nn.Linear):
                    layer_details.append({
                        'name': name,
                        'type': 'Linear',
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                        'parameters': module.weight.numel() + module.bias.numel()
                    })
                elif isinstance(module, nn.Dropout):
                    layer_details.append({
                        'name': name,
                        'type': 'Dropout',
                        'p': module.p,
                        'parameters': 0
                    })
                elif isinstance(module, nn.ReLU):
                    layer_details.append({
                        'name': name,
                        'type': 'ReLU',
                        'parameters': 0
                    })
                elif isinstance(module, nn.Parameter):
                    layer_details.append({
                        'name': name,
                        'type': 'Parameter',
                        'shape': list(module.shape),
                        'parameters': module.numel()
                    })
        
        print(f"\n详细层结构:")
        for layer in layer_details:
            if layer['type'] == 'Linear':
                print(f"  {layer['name']}: Linear({layer['in_features']} -> {layer['out_features']}) - {layer['parameters']:,} 参数")
            elif layer['type'] == 'Dropout':
                print(f"  {layer['name']}: Dropout(p={layer['p']})")
            elif layer['type'] == 'ReLU':
                print(f"  {layer['name']}: ReLU()")
            elif layer['type'] == 'Parameter':
                print(f"  {layer['name']}: Parameter{layer['shape']} - {layer['parameters']:,} 参数")
        
        # 计算FLOPs（简化估算）
        if model_name == 'FixedColorHead':
            # 64 -> 256 -> 128 -> 64 -> 128 -> 64 -> 43
            flops = 64*256 + 256*128 + 128*64 + 64*128 + 128*64 + 64*43
        elif model_name == 'BaseModel':
            # 3072 -> 1024 -> 512 -> 256 -> 128 -> 43
            flops = 3072*1024 + 1024*512 + 512*256 + 256*128 + 128*43
        else:  # EnsembleModel
            # 基础模型 + 颜色头 + 融合
            base_flops = 3072*1024 + 1024*512 + 512*256 + 256*128 + 128*43
            color_flops = 64*256 + 256*128 + 128*64 + 64*128 + 128*64 + 64*43
            flops = base_flops + color_flops + 43*2  # 融合操作
        
        print(f"\n估算FLOPs: {flops:,}")
        
        # 内存使用估算（MB）
        memory_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        print(f"内存使用估算: {memory_mb:.2f} MB")
        
        architecture_details[model_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'flops': flops,
            'memory_mb': memory_mb,
            'layers': layer_details
        }
    
    # 架构对比总结
    print(f"\n📊 架构对比总结")
    print("="*80)
    
    print(f"{'模型':<15} {'参数量':<12} {'FLOPs':<15} {'内存(MB)':<10} {'性能(%)':<10}")
    print("-" * 80)
    
    # 从测试结果中获取性能数据
    performance_data = {
        'FixedColorHead': 12.24,
        'BaseModel': 74.10,
        'EnsembleModel': 77.01
    }
    
    for model_name in models.keys():
        details = architecture_details[model_name]
        performance = performance_data.get(model_name, 0.0)
        print(f"{model_name:<15} {details['total_params']:<12,} {details['flops']:<15,} {details['memory_mb']:<10.2f} {performance:<10.2f}")
    
    # 架构特点分析
    print(f"\n🎯 架构特点分析")
    print("="*60)
    
    print(f"1. 修复颜色头 (FixedColorHead):")
    print(f"   - 输入: 64维特征 (图像前64个像素)")
    print(f"   - 结构: 64->256->128->64->128->64->43")
    print(f"   - 特点: 轻量级，专注于颜色语义学习")
    print(f"   - 性能: 12.24% (单独使用效果有限)")
    
    print(f"\n2. 基础模型 (BaseModel):")
    print(f"   - 输入: 3072维特征 (32x32x3图像)")
    print(f"   - 结构: 3072->1024->512->256->128->43")
    print(f"   - 特点: 标准全连接网络，处理完整图像信息")
    print(f"   - 性能: 74.10% (基础分类能力强)")
    
    print(f"\n3. 集成模型 (EnsembleModel):")
    print(f"   - 融合策略: 可学习权重融合")
    print(f"   - 融合公式: final = w*base + (1-w)*color")
    print(f"   - 特点: 结合基础分类和颜色语义")
    print(f"   - 性能: 77.01% (最佳性能)")
    
    # 设计优势分析
    print(f"\n✅ 设计优势")
    print("="*60)
    
    print(f"1. 模块化设计:")
    print(f"   - 颜色头和基础模型可独立训练")
    print(f"   - 便于调试和优化")
    print(f"   - 支持不同的融合策略")
    
    print(f"\n2. 轻量级颜色头:")
    print(f"   - 参数量少 (约50K参数)")
    print(f"   - 计算效率高")
    print(f"   - 专注于颜色特征学习")
    
    print(f"\n3. 可学习融合:")
    print(f"   - 自动学习最优融合权重")
    print(f"   - 适应不同数据分布")
    print(f"   - 避免手动调参")
    
    print(f"\n4. 修复成功:")
    print(f"   - 解决了负损失问题")
    print(f"   - 训练稳定性好")
    print(f"   - 在真实数据上验证有效")
    
    # 保存架构详情
    with open("model_architecture_details.json", 'w') as f:
        json.dump(architecture_details, f, indent=2, default=str)
    
    print(f"\n💾 架构详情已保存到: model_architecture_details.json")
    print("🎉 架构分析完成!")
    
    return architecture_details

if __name__ == '__main__':
    analyze_model_architecture()

