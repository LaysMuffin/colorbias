import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class ColorRelationReasoning(nn.Module):
    """颜色关系推理模块"""
    
    def __init__(self, color_dim=8, relation_dim=64, num_relations=8):
        super().__init__()
        
        # 颜色关系定义
        self.color_relations = {
            'complementary': {  # 互补色关系
                'red': 'green',
                'green': 'red', 
                'blue': 'orange',
                'orange': 'blue',
                'yellow': 'purple',
                'purple': 'yellow'
            },
            'analogous': {  # 类似色关系
                'red': ['orange', 'purple'],
                'orange': ['red', 'yellow'],
                'yellow': ['orange', 'green'],
                'green': ['yellow', 'blue'],
                'blue': ['green', 'purple'],
                'purple': ['blue', 'red']
            },
            'triadic': {  # 三角色关系
                'red': ['yellow', 'blue'],
                'yellow': ['red', 'blue'],
                'blue': ['red', 'yellow'],
                'green': ['orange', 'purple'],
                'orange': ['green', 'purple'],
                'purple': ['green', 'orange']
            },
            'contrast': {  # 对比色关系
                'red': ['white', 'black'],
                'blue': ['white', 'black'],
                'yellow': ['black'],
                'green': ['white', 'black'],
                'orange': ['white', 'black'],
                'purple': ['white', 'black'],
                'white': ['red', 'blue', 'green', 'orange', 'purple'],
                'black': ['red', 'blue', 'yellow', 'green', 'orange', 'purple']
            }
        }
        
        # 颜色到ID的映射
        self.color_to_id = {
            'red': 0, 'orange': 1, 'yellow': 2, 'green': 3,
            'blue': 4, 'purple': 5, 'white': 6, 'black': 7
        }
        
        # 确保color_dim与颜色数量匹配
        self.color_dim = max(self.color_to_id.values()) + 1
        
        self.id_to_color = {v: k for k, v in self.color_to_id.items()}
        
        # 关系推理网络
        self.relation_encoder = nn.Sequential(
            nn.Linear(self.color_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 关系分类器
        self.relation_classifier = nn.Sequential(
            nn.Linear(relation_dim, relation_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(relation_dim // 2, num_relations)
        )
        
        # 关系嵌入
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        
        # 颜色组合推理器
        self.combination_reasoner = nn.Sequential(
            nn.Linear(relation_dim * 3, relation_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(relation_dim, 5)  # 5种颜色组合
        )
        
        # 关系类型定义
        self.relation_types = [
            'complementary', 'analogous', 'triadic', 'contrast',
            'same', 'different', 'harmonious', 'conflicting'
        ]
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def encode_color_pair(self, color1_features: torch.Tensor, color2_features: torch.Tensor) -> torch.Tensor:
        """编码颜色对"""
        # 连接两个颜色特征
        combined = torch.cat([color1_features, color2_features], dim=-1)
        # 编码关系
        relation_features = self.relation_encoder(combined)
        return relation_features
    
    def predict_relation(self, relation_features: torch.Tensor) -> torch.Tensor:
        """预测颜色关系类型"""
        return self.relation_classifier(relation_features)
    
    def get_ground_truth_relation(self, color1: str, color2: str) -> int:
        """获取真实的关系类型"""
        if color1 == color2:
            return 4  # 'same'
        
        # 检查互补色关系
        if color2 in self.color_relations['complementary'].get(color1, []):
            return 0  # 'complementary'
        
        # 检查类似色关系
        if color2 in self.color_relations['analogous'].get(color1, []):
            return 1  # 'analogous'
        
        # 检查三角色关系
        if color2 in self.color_relations['triadic'].get(color1, []):
            return 2  # 'triadic'
        
        # 检查对比色关系
        if color2 in self.color_relations['contrast'].get(color1, []):
            return 3  # 'contrast'
        
        # 检查和谐关系（在同一个组合中）
        harmonious_combinations = [
            ['red', 'white', 'black'],
            ['blue', 'white'],
            ['yellow', 'white', 'black'],
            ['red', 'white'],
            ['black', 'white']
        ]
        
        for combo in harmonious_combinations:
            if color1 in combo and color2 in combo:
                return 6  # 'harmonious'
        
        return 7  # 'conflicting'
    
    def reason_color_combination(self, color_features: List[torch.Tensor]) -> torch.Tensor:
        """推理颜色组合"""
        batch_size = color_features[0].size(0)
        
        if len(color_features) < 2:
            return torch.zeros(batch_size, 5, device=color_features[0].device)
        
        # 计算所有颜色对的关系
        relation_features_list = []
        
        for i in range(len(color_features)):
            for j in range(i + 1, len(color_features)):
                relation_features = self.encode_color_pair(
                    color_features[i], color_features[j]
                )
                relation_features_list.append(relation_features)
        
        if not relation_features_list:
            return torch.zeros(batch_size, 5, device=color_features[0].device)
        
        # 平均所有关系特征
        avg_relation_features = torch.stack(relation_features_list).mean(dim=0)
        
        # 推理颜色组合
        combination_logits = self.combination_reasoner(avg_relation_features)
        return combination_logits
    
    def forward(self, color_features: torch.Tensor, color_names: List[str] = None) -> Dict:
        """前向传播"""
        batch_size = color_features.size(0)
        
        # 如果没有提供颜色名称，使用默认的
        if color_names is None:
            color_names = ['red', 'white', 'black'] * (batch_size // 3 + 1)
            color_names = color_names[:batch_size]
        
        # 编码颜色对关系
        color_dim = self.color_dim
        relation_features = self.encode_color_pair(
            color_features[:, :color_dim],  # 第一个颜色
            color_features[:, color_dim:color_dim*2] if color_features.size(1) >= color_dim*2 else color_features[:, :color_dim]  # 第二个颜色
        )
        
        # 预测关系类型
        relation_logits = self.predict_relation(relation_features)
        relation_probs = F.softmax(relation_logits, dim=-1)
        
        # 推理颜色组合
        combination_logits = self.reason_color_combination([color_features])
        combination_logits = combination_logits.to(color_features.device)
        combination_probs = F.softmax(combination_logits, dim=-1)
        
        return {
            'relation_features': relation_features,
            'relation_logits': relation_logits,
            'relation_probs': relation_probs,
            'combination_logits': combination_logits,
            'combination_probs': combination_probs
        }

class ColorRelationLoss(nn.Module):
    """颜色关系损失函数"""
    
    def __init__(self, relation_weight=0.1, combination_weight=0.2):
        super().__init__()
        self.relation_weight = relation_weight
        self.combination_weight = combination_weight
        
    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        """计算颜色关系损失"""
        # 确保设备一致
        device = outputs['relation_logits'].device
        relation_labels = targets['relation_labels'].to(device)
        combination_labels = targets['combination_labels'].to(device)
        
        # 关系分类损失
        relation_loss = F.cross_entropy(
            outputs['relation_logits'], 
            relation_labels,
            label_smoothing=0.1
        )
        
        # 组合推理损失
        combination_loss = F.cross_entropy(
            outputs['combination_logits'],
            combination_labels,
            label_smoothing=0.1
        )
        
        # 关系一致性损失
        consistency_loss = self._compute_consistency_loss(
            outputs['relation_probs'],
            targets['relation_labels']
        )
        
        # 总损失
        total_loss = (relation_loss + 
                     self.combination_weight * combination_loss +
                     self.relation_weight * consistency_loss)
        
        return {
            'total_loss': total_loss,
            'relation_loss': relation_loss,
            'combination_loss': combination_loss,
            'consistency_loss': consistency_loss
        }
    
    def _compute_consistency_loss(self, relation_probs: torch.Tensor, relation_labels: torch.Tensor) -> torch.Tensor:
        """计算关系一致性损失"""
        # 鼓励预测的关系概率分布更加确定
        entropy = -torch.sum(relation_probs * torch.log(relation_probs + 1e-8), dim=-1)
        consistency_loss = torch.mean(entropy)
        return consistency_loss

class ColorRelationDataset:
    """颜色关系数据集"""
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.color_relations = ColorRelationReasoning()
        self.data = []
        self.labels = []
        
        self._generate_data()
    
    def _generate_data(self):
        """生成颜色关系数据"""
        colors = list(self.color_relations.color_to_id.keys())
        
        for _ in range(self.num_samples):
            # 随机选择两个颜色
            color1, color2 = np.random.choice(colors, 2, replace=False)
            
            # 生成颜色特征（简化版本）
            color1_features = self._generate_color_features(color1)
            color2_features = self._generate_color_features(color2)
            
            # 获取真实关系
            relation_label = self.color_relations.get_ground_truth_relation(color1, color2)
            
            # 确定颜色组合标签
            combination_label = self._get_combination_label([color1, color2])
            
            self.data.append({
                'color1': color1,
                'color2': color2,
                'color1_features': color1_features,
                'color2_features': color2_features,
                'relation_label': relation_label,
                'combination_label': combination_label
            })
    
    def _generate_color_features(self, color: str) -> torch.Tensor:
        """生成颜色特征"""
        # 简化的颜色特征生成
        features = torch.zeros(self.color_relations.color_dim)
        color_id = self.color_relations.color_to_id.get(color, 0)
        features[color_id] = 1.0
        
        # 添加一些噪声
        noise = torch.randn(self.color_relations.color_dim) * 0.1
        features = features + noise
        features = torch.clamp(features, 0, 1)
        
        return features
    
    def _get_combination_label(self, colors: List[str]) -> int:
        """获取颜色组合标签"""
        # 根据颜色组合定义确定标签
        combinations = [
            ['red', 'white', 'black'],
            ['blue', 'white'],
            ['yellow', 'white', 'black'],
            ['red', 'white'],
            ['black', 'white']
        ]
        
        for i, combo in enumerate(combinations):
            if all(color in combo for color in colors):
                return i
        
        return 0  # 默认标签
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 合并颜色特征
        combined_features = torch.cat([
            item['color1_features'],
            item['color2_features']
        ])
        
        return {
            'features': combined_features,
            'relation_label': item['relation_label'],
            'combination_label': item['combination_label'],
            'color1': item['color1'],
            'color2': item['color2']
        }

def train_color_relation_reasoning(model: ColorRelationReasoning, 
                                 dataset: ColorRelationDataset,
                                 epochs: int = 50,
                                 lr: float = 0.001) -> Dict:
    """训练颜色关系推理模型"""
    print("🎨 开始训练颜色关系推理模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = ColorRelationLoss()
    
    # 训练历史
    history = {
        'relation_loss': [],
        'combination_loss': [],
        'consistency_loss': [],
        'total_loss': [],
        'relation_accuracy': [],
        'combination_accuracy': []
    }
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        relation_correct = 0
        combination_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            relation_labels = batch['relation_label'].to(device)
            combination_labels = batch['combination_label'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features)
            
            # 计算损失
            targets = {
                'relation_labels': relation_labels,
                'combination_labels': combination_labels
            }
            loss_dict = criterion(outputs, targets)
            
            # 反向传播
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            total_loss += loss_dict['total_loss'].item()
            
            # 计算准确率
            _, relation_pred = outputs['relation_logits'].max(1)
            _, combination_pred = outputs['combination_logits'].max(1)
            
            relation_correct += relation_pred.eq(relation_labels).sum().item()
            combination_correct += combination_pred.eq(combination_labels).sum().item()
            total_samples += features.size(0)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        avg_loss = total_loss / len(train_loader)
        relation_accuracy = 100.0 * relation_correct / total_samples
        combination_accuracy = 100.0 * combination_correct / total_samples
        
        history['total_loss'].append(avg_loss)
        history['relation_accuracy'].append(relation_accuracy)
        history['combination_accuracy'].append(combination_accuracy)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss: {avg_loss:.4f}, "
                  f"Relation Acc: {relation_accuracy:.2f}%, "
                  f"Combination Acc: {combination_accuracy:.2f}%")
        
        # 保存最佳模型
        if combination_accuracy > best_accuracy:
            best_accuracy = combination_accuracy
            torch.save(model.state_dict(), 'best_color_relation_model.pth')
    
    print(f"✅ 训练完成！最佳组合准确率: {best_accuracy:.2f}%")
    return history

def test_color_relation_reasoning(model: ColorRelationReasoning):
    """测试颜色关系推理"""
    print("🧪 测试颜色关系推理...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 测试用例
    test_cases = [
        ('red', 'white'),
        ('blue', 'white'),
        ('yellow', 'black'),
        ('red', 'green'),
        ('blue', 'orange'),
        ('white', 'black')
    ]
    
    results = []
    
    with torch.no_grad():
        for color1, color2 in test_cases:
            # 生成特征
            color1_features = torch.zeros(model.color_dim)
            color2_features = torch.zeros(model.color_dim)
            
            color1_features[model.color_to_id[color1]] = 1.0
            color2_features[model.color_to_id[color2]] = 1.0
            
            combined_features = torch.cat([color1_features, color2_features]).unsqueeze(0).to(device)
            
            # 推理
            outputs = model(combined_features, [color1, color2])
            
            # 获取预测结果
            relation_pred = outputs['relation_probs'].argmax(1).item()
            combination_pred = outputs['combination_probs'].argmax(1).item()
            
            # 获取真实关系
            true_relation = model.get_ground_truth_relation(color1, color2)
            
            result = {
                'color1': color1,
                'color2': color2,
                'predicted_relation': model.relation_types[relation_pred],
                'true_relation': model.relation_types[true_relation],
                'relation_correct': relation_pred == true_relation,
                'combination_pred': combination_pred
            }
            
            results.append(result)
            
            print(f"{color1} + {color2}: "
                  f"预测关系: {result['predicted_relation']}, "
                  f"真实关系: {result['true_relation']}, "
                  f"正确: {result['relation_correct']}")
    
    # 计算准确率
    relation_accuracy = sum(r['relation_correct'] for r in results) / len(results) * 100
    print(f"🎯 关系推理准确率: {relation_accuracy:.2f}%")
    
    return results

if __name__ == "__main__":
    # 创建模型
    model = ColorRelationReasoning()
    
    # 创建数据集
    dataset = ColorRelationDataset(num_samples=2000)
    
    # 训练模型
    history = train_color_relation_reasoning(model, dataset, epochs=50)
    
    # 测试模型
    results = test_color_relation_reasoning(model)
    
    # 保存结果
    with open('color_relation_results.json', 'w') as f:
        json.dump({
            'history': history,
            'test_results': results
        }, f, indent=2)
    
    print("✅ 颜色关系推理模块训练和测试完成！")
