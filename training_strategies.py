# training_strategies.py
# 训练策略模块 - 分阶段训练和课程学习

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple, Callable
import time

class StagedTraining:
    """分阶段训练策略"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 训练阶段配置
        self.stages = {
            'color_detector': {
                'epochs': 5,
                'lr': 0.001,
                'description': '预训练颜色检测器'
            },
            'shape_decorrelation': {
                'epochs': 5,
                'lr': 0.0005,
                'description': '训练形状去相关'
            },
            'joint_training': {
                'epochs': 10,
                'lr': 0.0001,
                'description': '联合训练'
            }
        }
        
        # 训练历史
        self.training_history = {
            'stages': {},
            'best_val_acc': 0,
            'best_epoch': 0
        }
    
    def train_color_detector_only(self, epochs: int, lr: float):
        """阶段1: 只训练颜色检测器"""
        print(f"🎨 阶段1: 预训练颜色检测器 ({epochs} epochs, lr={lr})")
        
        # 冻结其他组件
        for name, param in self.model.named_parameters():
            if 'color_detector' not in name and 'color_feature_extractor' not in name:
                param.requires_grad = False
        
        # 优化器
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 只计算颜色检测损失
                if hasattr(self.model, 'compute_enhanced_loss'):
                    loss_dict = self.model.compute_enhanced_loss(outputs, target, data)
                    loss = loss_dict['color_consistency_loss'] + loss_dict['color_invariance_loss']
                else:
                    loss = nn.CrossEntropyLoss()(outputs['color_logits'], target)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 验证
            val_acc = self._validate_model()
            print(f"  Epoch {epoch+1} - Avg Loss: {total_loss/len(self.train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
    
    def train_shape_decorrelation(self, epochs: int, lr: float):
        """阶段2: 训练形状去相关"""
        print(f"🔄 阶段2: 训练形状去相关 ({epochs} epochs, lr={lr})")
        
        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 主要关注形状去相关损失
                if hasattr(self.model, 'compute_enhanced_loss'):
                    loss_dict = self.model.compute_enhanced_loss(outputs, target, data)
                    loss = (
                        loss_dict['shape_decorr_loss'] * 2.0 +  # 增加权重
                        loss_dict['color_consistency_loss'] * 0.5 +
                        loss_dict['ce_loss'] * 0.5
                    )
                else:
                    loss = nn.CrossEntropyLoss()(outputs['logits'], target)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 验证
            val_acc = self._validate_model()
            print(f"  Epoch {epoch+1} - Avg Loss: {total_loss/len(self.train_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    def train_joint_model(self, epochs: int, lr: float):
        """阶段3: 联合训练"""
        print(f"🤝 阶段3: 联合训练 ({epochs} epochs, lr={lr})")
        
        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        best_val_acc = 0
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 完整损失
                if hasattr(self.model, 'compute_enhanced_loss'):
                    loss_dict = self.model.compute_enhanced_loss(outputs, target, data)
                    loss = loss_dict['total_loss']
                else:
                    loss = nn.CrossEntropyLoss()(outputs['logits'], target)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 更新学习率
            scheduler.step()
            
            # 验证
            val_acc = self._validate_model()
            print(f"  Epoch {epoch+1} - Avg Loss: {total_loss/len(self.train_loader):.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'checkpoints/best_staged_model_epoch_{epoch+1}.pth')
        
        return best_val_acc
    
    def _validate_model(self) -> float:
        """验证模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                
                if 'final_logits' in outputs:
                    predictions = torch.argmax(outputs['final_logits'], dim=1)
                elif 'logits' in outputs:
                    predictions = torch.argmax(outputs['logits'], dim=1)
                else:
                    predictions = torch.argmax(outputs, dim=1)
                
                correct += (predictions == target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def run_staged_training(self):
        """运行完整的分阶段训练"""
        print("🚀 开始分阶段训练")
        print("="*60)
        
        start_time = time.time()
        
        for stage_name, config in self.stages.items():
            print(f"\n📋 {config['description']}")
            print("-" * 40)
            
            if stage_name == 'color_detector':
                self.train_color_detector_only(config['epochs'], config['lr'])
            elif stage_name == 'shape_decorrelation':
                self.train_shape_decorrelation(config['epochs'], config['lr'])
            elif stage_name == 'joint_training':
                best_acc = self.train_joint_model(config['epochs'], config['lr'])
                self.training_history['best_val_acc'] = best_acc
        
        total_time = time.time() - start_time
        print(f"\n🎉 分阶段训练完成! 总时间: {total_time/60:.2f}分钟")
        print(f"最佳验证准确率: {self.training_history['best_val_acc']:.4f}")

class CurriculumLearning:
    """课程学习策略"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 课程配置
        self.curriculum = {
            'single_color': {
                'classes': [14, 17, 33, 34, 35],  # Stop, No Entry, Turn Right, Turn Left, Ahead Only
                'epochs': 3,
                'description': '单色标志学习'
            },
            'dual_color': {
                'classes': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11],  # Speed limits
                'epochs': 3,
                'description': '双色标志学习'
            },
            'multi_color': {
                'classes': [12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],  # 复杂标志
                'epochs': 3,
                'description': '多色标志学习'
            },
            'all_classes': {
                'classes': list(range(43)),  # 所有类别
                'epochs': 10,
                'description': '全类别训练'
            }
        }
        
        # 训练历史
        self.training_history = {
            'curriculum': {},
            'best_val_acc': 0,
            'best_epoch': 0
        }
    
    def create_subset_loader(self, class_indices: List[int]) -> DataLoader:
        """创建特定类别的数据加载器"""
        # 获取指定类别的样本索引
        subset_indices = []
        for idx, (_, target) in enumerate(self.train_loader.dataset):
            if target in class_indices:
                subset_indices.append(idx)
        
        # 创建子集
        subset = Subset(self.train_loader.dataset, subset_indices)
        subset_loader = DataLoader(
            subset, 
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=self.train_loader.num_workers
        )
        
        return subset_loader
    
    def train_on_subset(self, class_indices: List[int], epochs: int, lr: float, stage_name: str):
        """在特定类别子集上训练"""
        print(f"📚 {stage_name}: 训练类别 {class_indices} ({epochs} epochs, lr={lr})")
        
        # 创建子集加载器
        subset_loader = self.create_subset_loader(class_indices)
        print(f"  子集大小: {len(subset_loader.dataset)} 样本")
        
        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(subset_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 计算损失
                if hasattr(self.model, 'compute_enhanced_loss'):
                    loss_dict = self.model.compute_enhanced_loss(outputs, target, data)
                    loss = loss_dict['total_loss']
                else:
                    loss = nn.CrossEntropyLoss()(outputs['logits'], target)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 计算准确率
                if 'final_logits' in outputs:
                    predictions = torch.argmax(outputs['final_logits'], dim=1)
                elif 'logits' in outputs:
                    predictions = torch.argmax(outputs['logits'], dim=1)
                else:
                    predictions = torch.argmax(outputs, dim=1)
                
                correct += (predictions == target).sum().item()
                total += target.size(0)
                
                if batch_idx % 20 == 0:
                    print(f"    Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 计算子集准确率
            subset_acc = correct / total
            avg_loss = total_loss / len(subset_loader)
            print(f"    Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Subset Acc: {subset_acc:.4f}")
            
            # 在完整验证集上验证
            val_acc = self._validate_model()
            print(f"    Epoch {epoch+1} - Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > self.training_history['best_val_acc']:
                self.training_history['best_val_acc'] = val_acc
                torch.save(self.model.state_dict(), f'checkpoints/best_curriculum_{stage_name}_epoch_{epoch+1}.pth')
    
    def _validate_model(self) -> float:
        """验证模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                
                if 'final_logits' in outputs:
                    predictions = torch.argmax(outputs['final_logits'], dim=1)
                elif 'logits' in outputs:
                    predictions = torch.argmax(outputs['logits'], dim=1)
                else:
                    predictions = torch.argmax(outputs, dim=1)
                
                correct += (predictions == target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def run_curriculum_learning(self):
        """运行课程学习"""
        print("📚 开始课程学习")
        print("="*60)
        
        start_time = time.time()
        
        for stage_name, config in self.curriculum.items():
            print(f"\n📋 {config['description']}")
            print("-" * 40)
            
            self.train_on_subset(
                config['classes'], 
                config['epochs'], 
                0.001,  # 固定学习率
                stage_name
            )
        
        total_time = time.time() - start_time
        print(f"\n🎉 课程学习完成! 总时间: {total_time/60:.2f}分钟")
        print(f"最佳验证准确率: {self.training_history['best_val_acc']:.4f}")

def test_training_strategies():
    """测试训练策略模块"""
    print("🎯 测试训练策略模块")
    print("="*60)
    
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
    
    # 创建测试数据
    class TestDataset:
        def __init__(self, size=1000):
            self.size = size
            self.data = torch.randn(size, 64)
            self.targets = torch.randint(0, 43, (size,))
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
        
        def __len__(self):
            return self.size
    
    # 创建数据加载器
    dataset = TestDataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestModel().to(device)
    
    print(f"使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试分阶段训练
    print(f"\n🧪 测试分阶段训练策略:")
    staged_trainer = StagedTraining(model, train_loader, val_loader, device)
    
    # 测试课程学习
    print(f"\n🧪 测试课程学习策略:")
    curriculum_trainer = CurriculumLearning(model, train_loader, val_loader, device)
    
    print(f"\n✅ 训练策略模块测试完成")

if __name__ == '__main__':
    test_training_strategies()
