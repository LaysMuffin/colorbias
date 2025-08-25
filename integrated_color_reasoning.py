import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

# 导入现有模块
from improved_color_combination import ImprovedColorCombinationHead
from color_relation_reasoning import ColorRelationReasoning

class IntegratedColorReasoningModel(nn.Module):
    """集成颜色推理模型"""
    
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 主要颜色组合分类器
        self.color_classifier = ImprovedColorCombinationHead(num_classes=num_classes)
        
        # 颜色关系推理模块
        self.relation_reasoner = ColorRelationReasoning()
        
        # 集成融合层
        self.integration_layer = nn.Sequential(
            nn.Linear(num_classes + 5, 64),  # 分类器输出 + 关系推理输出
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        # 注意力权重
        self.classifier_weight = nn.Parameter(torch.tensor(0.7))
        self.reasoner_weight = nn.Parameter(torch.tensor(0.3))
        
        # 关系一致性损失权重
        self.relation_consistency_weight = 0.1
        
    def forward(self, x: torch.Tensor) -> Dict:
        """前向传播"""
        # 1. 主要颜色分类器
        classifier_outputs = self.color_classifier(x)
        classifier_logits = classifier_outputs['combination_logits']
        classifier_probs = F.softmax(classifier_logits, dim=-1)
        
        # 2. 颜色关系推理
        # 提取颜色特征用于关系推理
        color_features = self._extract_color_features(x)
        relation_outputs = self.relation_reasoner(color_features)
        relation_logits = relation_outputs['combination_logits']
        relation_probs = F.softmax(relation_logits, dim=-1)
        
        # 3. 集成融合
        combined_features = torch.cat([classifier_probs, relation_probs], dim=-1)
        integrated_logits = self.integration_layer(combined_features)
        integrated_probs = F.softmax(integrated_logits, dim=-1)
        
        # 4. 加权融合
        final_logits = (self.classifier_weight * classifier_logits + 
                       self.reasoner_weight * relation_logits)
        final_probs = F.softmax(final_logits, dim=-1)
        
        return {
            'classifier_logits': classifier_logits,
            'classifier_probs': classifier_probs,
            'relation_logits': relation_logits,
            'relation_probs': relation_probs,
            'integrated_logits': integrated_logits,
            'integrated_probs': integrated_probs,
            'final_logits': final_logits,
            'final_probs': final_probs,
            'relation_features': relation_outputs['relation_features'],
            'color_semantic': classifier_outputs['color_semantic']
        }
    
    def _extract_color_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取颜色特征用于关系推理"""
        batch_size = x.size(0)
        
        # 简化的颜色特征提取
        # 计算每个通道的平均值作为颜色特征
        color_features = torch.mean(x, dim=(2, 3))  # [batch, 3]
        
        # 扩展到8维（对应8种颜色）
        expanded_features = torch.zeros(batch_size, 8, device=x.device)
        
        # 基于RGB值映射到颜色特征
        for i in range(batch_size):
            r, g, b = color_features[i]
            
            # 简单的颜色映射逻辑
            if r > 0.5 and g < 0.3 and b < 0.3:
                expanded_features[i, 0] = 1.0  # red
            elif r > 0.5 and g > 0.3 and b < 0.3:
                expanded_features[i, 1] = 1.0  # orange
            elif r > 0.5 and g > 0.5 and b < 0.3:
                expanded_features[i, 2] = 1.0  # yellow
            elif r < 0.3 and g > 0.5 and b < 0.3:
                expanded_features[i, 3] = 1.0  # green
            elif r < 0.3 and g < 0.3 and b > 0.5:
                expanded_features[i, 4] = 1.0  # blue
            elif r > 0.5 and g < 0.3 and b > 0.5:
                expanded_features[i, 5] = 1.0  # purple
            elif r > 0.7 and g > 0.7 and b > 0.7:
                expanded_features[i, 6] = 1.0  # white
            elif r < 0.3 and g < 0.3 and b < 0.3:
                expanded_features[i, 7] = 1.0  # black
            else:
                # 默认映射到主要颜色
                max_val, max_idx = torch.max(color_features[i], 0)
                expanded_features[i, max_idx] = 1.0
        
        return expanded_features
    
    def compute_integrated_loss(self, outputs: Dict, targets: torch.Tensor) -> Dict:
        """计算集成损失"""
        # 1. 分类器损失
        classifier_loss = F.cross_entropy(
            outputs['classifier_logits'], 
            targets, 
            label_smoothing=0.1
        )
        
        # 2. 关系推理损失
        relation_loss = F.cross_entropy(
            outputs['relation_logits'],
            targets,
            label_smoothing=0.1
        )
        
        # 3. 集成损失
        integrated_loss = F.cross_entropy(
            outputs['integrated_logits'],
            targets,
            label_smoothing=0.1
        )
        
        # 4. 最终融合损失
        final_loss = F.cross_entropy(
            outputs['final_logits'],
            targets,
            label_smoothing=0.1
        )
        
        # 5. 关系一致性损失
        consistency_loss = self._compute_relation_consistency_loss(outputs)
        
        # 6. 总损失
        total_loss = (classifier_loss + 
                     0.5 * relation_loss + 
                     0.3 * integrated_loss + 
                     final_loss + 
                     self.relation_consistency_weight * consistency_loss)
        
        return {
            'total_loss': total_loss,
            'classifier_loss': classifier_loss,
            'relation_loss': relation_loss,
            'integrated_loss': integrated_loss,
            'final_loss': final_loss,
            'consistency_loss': consistency_loss
        }
    
    def _compute_relation_consistency_loss(self, outputs: Dict) -> torch.Tensor:
        """计算关系一致性损失"""
        # 鼓励分类器和关系推理器的一致性
        classifier_probs = outputs['classifier_probs']
        relation_probs = outputs['relation_probs']
        
        # KL散度损失
        kl_loss = F.kl_div(
            torch.log(relation_probs + 1e-8),
            classifier_probs,
            reduction='batchmean'
        )
        
        return kl_loss

def train_integrated_model(model: IntegratedColorReasoningModel,
                         train_loader,
                         val_loader,
                         epochs: int = 30,
                         lr: float = 0.0005) -> Dict:
    """训练集成模型"""
    print("🎨 开始训练集成颜色推理模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'classifier_accuracy': [],
        'relation_accuracy': [],
        'integrated_accuracy': []
    }
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            loss_dict = model.compute_integrated_loss(outputs, targets)
            
            # 反向传播
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            train_loss += loss_dict['total_loss'].item()
            
            # 计算准确率
            _, predicted = outputs['final_logits'].max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 30 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss_dict['total_loss'].item():.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        classifier_correct = 0
        relation_correct = 0
        integrated_correct = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                loss_dict = model.compute_integrated_loss(outputs, targets)
                
                val_loss += loss_dict['total_loss'].item()
                
                # 各种准确率
                _, final_pred = outputs['final_logits'].max(1)
                _, classifier_pred = outputs['classifier_logits'].max(1)
                _, relation_pred = outputs['relation_logits'].max(1)
                _, integrated_pred = outputs['integrated_logits'].max(1)
                
                val_total += targets.size(0)
                val_correct += final_pred.eq(targets).sum().item()
                classifier_correct += classifier_pred.eq(targets).sum().item()
                relation_correct += relation_pred.eq(targets).sum().item()
                integrated_correct += integrated_pred.eq(targets).sum().item()
        
        # 更新学习率
        scheduler.step()
        
        # 计算准确率
        train_accuracy = 100.0 * train_correct / train_total
        val_accuracy = 100.0 * val_correct / val_total
        classifier_accuracy = 100.0 * classifier_correct / val_total
        relation_accuracy = 100.0 * relation_correct / val_total
        integrated_accuracy = 100.0 * integrated_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['classifier_accuracy'].append(classifier_accuracy)
        history['relation_accuracy'].append(relation_accuracy)
        history['integrated_accuracy'].append(integrated_accuracy)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Acc: {train_accuracy:.2f}%, "
                  f"Val Acc: {val_accuracy:.2f}%, "
                  f"Classifier: {classifier_accuracy:.2f}%, "
                  f"Relation: {relation_accuracy:.2f}%, "
                  f"Integrated: {integrated_accuracy:.2f}%")
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_integrated_color_model.pth')
    
    print(f"✅ 训练完成！最佳验证准确率: {best_accuracy:.2f}%")
    return history

def test_integrated_model(model: IntegratedColorReasoningModel, test_loader):
    """测试集成模型"""
    print("🧪 测试集成颜色推理模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {
        'final_accuracy': 0.0,
        'classifier_accuracy': 0.0,
        'relation_accuracy': 0.0,
        'integrated_accuracy': 0.0,
        'detailed_results': []
    }
    
    final_correct = 0
    classifier_correct = 0
    relation_correct = 0
    integrated_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            
            # 计算各种准确率
            _, final_pred = outputs['final_logits'].max(1)
            _, classifier_pred = outputs['classifier_logits'].max(1)
            _, relation_pred = outputs['relation_logits'].max(1)
            _, integrated_pred = outputs['integrated_logits'].max(1)
            
            total_samples += targets.size(0)
            final_correct += final_pred.eq(targets).sum().item()
            classifier_correct += classifier_pred.eq(targets).sum().item()
            relation_correct += relation_pred.eq(targets).sum().item()
            integrated_correct += integrated_pred.eq(targets).sum().item()
            
            # 记录详细结果
            for i in range(targets.size(0)):
                results['detailed_results'].append({
                    'target': targets[i].item(),
                    'final_pred': final_pred[i].item(),
                    'classifier_pred': classifier_pred[i].item(),
                    'relation_pred': relation_pred[i].item(),
                    'integrated_pred': integrated_pred[i].item(),
                    'final_correct': final_pred[i].eq(targets[i]).item(),
                    'classifier_correct': classifier_pred[i].eq(targets[i]).item(),
                    'relation_correct': relation_pred[i].eq(targets[i]).item(),
                    'integrated_correct': integrated_pred[i].eq(targets[i]).item()
                })
    
    # 计算最终准确率
    results['final_accuracy'] = 100.0 * final_correct / total_samples
    results['classifier_accuracy'] = 100.0 * classifier_correct / total_samples
    results['relation_accuracy'] = 100.0 * relation_correct / total_samples
    results['integrated_accuracy'] = 100.0 * integrated_correct / total_samples
    
    print(f"🎯 测试结果:")
    print(f"  最终准确率: {results['final_accuracy']:.2f}%")
    print(f"  分类器准确率: {results['classifier_accuracy']:.2f}%")
    print(f"  关系推理准确率: {results['relation_accuracy']:.2f}%")
    print(f"  集成准确率: {results['integrated_accuracy']:.2f}%")
    
    return results

if __name__ == "__main__":
    # 创建集成模型
    model = IntegratedColorReasoningModel(num_classes=5)
    
    # 加载预训练的分类器权重
    try:
        model.color_classifier.load_state_dict(
            torch.load('best_improved_color_combination_head.pth', map_location='cpu')
        )
        print("✅ 加载了预训练的分类器权重")
    except:
        print("⚠️ 未找到预训练权重，将从头开始训练")
    
    # 加载预训练的关系推理权重
    try:
        model.relation_reasoner.load_state_dict(
            torch.load('best_color_relation_model.pth', map_location='cpu')
        )
        print("✅ 加载了预训练的关系推理权重")
    except:
        print("⚠️ 未找到关系推理权重，将从头开始训练")
    
    # 创建数据集（这里需要实际的GTSRB数据加载器）
    from improved_color_combination import BalancedColorCombinationDataset
    
    # 创建数据集
    train_dataset = BalancedColorCombinationDataset(size=1000, use_real_data=True)
    val_dataset = BalancedColorCombinationDataset(size=200, use_real_data=True)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 训练模型
    history = train_integrated_model(model, train_loader, val_loader, epochs=20)
    
    # 测试模型
    results = test_integrated_model(model, val_loader)
    
    # 保存结果
    with open('integrated_color_reasoning_results.json', 'w') as f:
        json.dump({
            'history': history,
            'test_results': results
        }, f, indent=2)
    
    print("✅ 集成颜色推理模型训练和测试完成！")
