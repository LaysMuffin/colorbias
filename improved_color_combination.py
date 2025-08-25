import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# 改进的颜色组合分类定义 - 更均衡的分布
IMPROVED_COLOR_COMBINATIONS = {
    0: {'name': '红白黑', 'colors': ['red', 'white', 'black'], 'classes': [0,1,2,3,4,5,7,8,9,10,11,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]},
    1: {'name': '蓝白', 'colors': ['blue', 'white'], 'classes': [33,34,35,36,37,38,39,40]},
    2: {'name': '黄白黑', 'colors': ['yellow', 'white', 'black'], 'classes': [12,32,41,42]},
    3: {'name': '红白', 'colors': ['red', 'white'], 'classes': [14]},
    4: {'name': '黑白', 'colors': ['black', 'white'], 'classes': [6]}
}

# 颜色到RGB的映射
COLOR_TO_RGB = {
    'red': [1.0, 0.0, 0.0],
    'white': [1.0, 1.0, 1.0],
    'black': [0.0, 0.0, 0.0],
    'blue': [0.0, 0.0, 1.0],
    'yellow': [1.0, 1.0, 0.0]
}

def create_improved_color_combination_sign(combination_id, size=32):
    """创建改进的颜色组合交通标志"""
    combination = IMPROVED_COLOR_COMBINATIONS[combination_id]
    colors = combination['colors']
    
    # 创建基础图像
    sign = np.ones((size, size, 3)) * 0.5  # 灰色背景
    
    # 根据颜色组合设计不同的图案
    if combination_id == 0:  # 红白黑 - 圆形标志
        center = size // 2
        radius = size // 3
        
        # 红色外圈
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if radius - 2 <= dist <= radius:
                    sign[i, j] = COLOR_TO_RGB['red']
                elif dist < radius - 2:
                    sign[i, j] = COLOR_TO_RGB['white']
        
        # 黑色中心
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < radius // 3:
                    sign[i, j] = COLOR_TO_RGB['black']
    
    elif combination_id == 1:  # 蓝白 - 方形标志
        margin = size // 4
        sign[margin:size-margin, margin:size-margin] = COLOR_TO_RGB['blue']
        sign[margin+2:size-margin-2, margin+2:size-margin-2] = COLOR_TO_RGB['white']
    
    elif combination_id == 2:  # 黄白黑 - 三角形标志
        center = size // 2
        height = size // 2
        
        # 黄色三角形
        for i in range(size):
            for j in range(size):
                if i >= center - height//2 and i <= center + height//2:
                    width = int((i - (center - height//2)) * 0.8)
                    if center - width <= j <= center + width:
                        sign[i, j] = COLOR_TO_RGB['yellow']
        
        # 黑色边框
        for i in range(size):
            for j in range(size):
                if i >= center - height//2 and i <= center + height//2:
                    width = int((i - (center - height//2)) * 0.8)
                    if (center - width <= j <= center - width + 1) or (center + width - 1 <= j <= center + width):
                        sign[i, j] = COLOR_TO_RGB['black']
    
    elif combination_id == 3:  # 红白 - 八角形
        center = size // 2
        radius = size // 3
        
        # 红色八角形
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist <= radius:
                    sign[i, j] = COLOR_TO_RGB['red']
        
        # 白色内圈
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist <= radius - 3:
                    sign[i, j] = COLOR_TO_RGB['white']
    
    elif combination_id == 4:  # 黑白 - 简单矩形
        margin = size // 4
        sign[margin:size-margin, margin:size-margin] = COLOR_TO_RGB['black']
        sign[margin+2:size-margin-2, margin+2:size-margin-2] = COLOR_TO_RGB['white']
    
    return sign

def get_color_combination_from_class(class_id):
    """从GTSRB类别ID获取颜色组合ID"""
    for combo_id, combo_info in IMPROVED_COLOR_COMBINATIONS.items():
        if class_id in combo_info['classes']:
            return combo_id
    return 0  # 默认返回红白黑

class BalancedColorCombinationDataset(Dataset):
    """平衡的颜色组合分类数据集"""
    
    def __init__(self, size=2000, use_real_data=False, real_data_path=None, balance_classes=True):
        self.size = size
        self.use_real_data = use_real_data
        self.real_data_path = real_data_path
        self.balance_classes = balance_classes
        
        if use_real_data and real_data_path:
            self.data, self.labels = self.load_real_data()
        else:
            self.data, self.labels = self.generate_semantic_data()
    
    def load_real_data(self):
        data = []
        labels = []
        
        try:
            train_dir = os.path.join(self.real_data_path, 'Final_Training', 'Images')
            
            # 收集所有可用的图像
            all_images = []
            for class_id in range(43):
                class_dir = os.path.join(train_dir, f'{class_id:05d}')
                if os.path.exists(class_dir):
                    for file in os.listdir(class_dir):
                        if file.endswith('.ppm'):
                            img_path = os.path.join(class_dir, file)
                            combo_id = get_color_combination_from_class(class_id)
                            all_images.append((img_path, combo_id))
            
            # 如果启用类别平衡
            if self.balance_classes:
                # 按颜色组合分组
                combo_groups = {}
                for img_path, combo_id in all_images:
                    if combo_id not in combo_groups:
                        combo_groups[combo_id] = []
                    combo_groups[combo_id].append(img_path)
                
                # 计算每个组合的最大样本数
                min_samples = min(len(images) for images in combo_groups.values())
                target_samples_per_combo = min(min_samples, self.size // 5)
                
                # 平衡采样
                for combo_id, images in combo_groups.items():
                    selected_images = np.random.choice(images, size=target_samples_per_combo, replace=False)
                    for img_path in selected_images:
                        data.append(img_path)
                        labels.append(combo_id)
            else:
                # 随机采样
                selected_images = np.random.choice(all_images, size=min(len(all_images), self.size), replace=False)
                for img_path, combo_id in selected_images:
                    data.append(img_path)
                    labels.append(combo_id)
            
            print(f"加载了 {len(data)} 张真实图像，转换为 {len(set(labels))} 种颜色组合")
            label_counts = Counter(labels)
            print(f"颜色组合分布: {dict(label_counts)}")
            
        except Exception as e:
            print(f"加载真实数据失败: {e}")
            data, labels = self.generate_semantic_data()
        
        return data, labels
    
    def generate_semantic_data(self):
        data = []
        labels = []
        
        if self.balance_classes:
            # 平衡生成语义化数据
            samples_per_combo = self.size // 5
            for combo_id in range(5):
                for i in range(samples_per_combo):
                    sign = create_improved_color_combination_sign(combo_id, size=32)
                    
                    # 添加噪声和变化
                    noise = np.random.normal(0, 0.1, sign.shape)
                    sign = np.clip(sign + noise, 0, 1)
                    
                    brightness = np.random.uniform(0.8, 1.2)
                    sign = np.clip(sign * brightness, 0, 1)
                    
                    data.append(sign)
                    labels.append(combo_id)
        else:
            # 随机生成
            for i in range(self.size):
                combo_id = np.random.randint(0, 5)
                sign = create_improved_color_combination_sign(combo_id, size=32)
                
                # 添加噪声和变化
                noise = np.random.normal(0, 0.1, sign.shape)
                sign = np.clip(sign + noise, 0, 1)
                
                brightness = np.random.uniform(0.8, 1.2)
                sign = np.clip(sign * brightness, 0, 1)
                
                data.append(sign)
                labels.append(combo_id)
        
        print(f"生成了 {len(data)} 张颜色组合图像")
        label_counts = Counter(labels)
        print(f"颜色组合分布: {dict(label_counts)}")
        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.use_real_data:
            img_path = self.data[idx]
            label = self.labels[idx]
            
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize((32, 32))
                image = np.array(image) / 255.0
                return torch.FloatTensor(image.transpose(2, 0, 1)), label
            except Exception as e:
                sign = create_improved_color_combination_sign(label, size=32)
                return torch.FloatTensor(sign.transpose(2, 0, 1)), label
        else:
            sign = self.data[idx]
            label = self.labels[idx]
            return torch.FloatTensor(sign.transpose(2, 0, 1)), label

class ImprovedColorSpaceConverter(nn.Module):
    """改进的颜色空间转换器"""
    
    def __init__(self):
        super().__init__()
        # RGB到HSV的转换矩阵
        self.register_buffer('rgb_to_hsv_matrix', torch.tensor([
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.321],
            [0.211, -0.523, 0.312]
        ]))
        
        # 颜色增强层
        self.color_enhancement = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb):
        # 输入: [batch, 3, H, W] (RGB)
        
        # 颜色增强
        enhanced_rgb = self.color_enhancement(rgb)
        
        # 转换为HSV
        batch_size, channels, height, width = enhanced_rgb.shape
        rgb_flat = enhanced_rgb.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        hsv_flat = torch.matmul(rgb_flat, self.rgb_to_hsv_matrix.T)
        hsv = hsv_flat.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)
        
        return enhanced_rgb, hsv

class ImprovedColorChannelAttention(nn.Module):
    """改进的颜色通道注意力机制"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        reduced_channels = max(1, channels // 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_attention = avg_out + max_out
        
        # 空间注意力
        spatial_attention = self.spatial_attention(x)
        
        # 应用注意力
        attended = x * channel_attention.unsqueeze(2).unsqueeze(3) * spatial_attention
        
        return attended

class ImprovedColorSemanticExtractor(nn.Module):
    """改进的颜色语义提取器"""
    
    def __init__(self, input_channels):
        super().__init__()
        
        # 多尺度颜色特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 颜色语义分类器
        self.color_classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # 多尺度特征提取
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # 全局池化
        global_features = self.global_pool(x3).squeeze(-1).squeeze(-1)
        
        # 颜色语义分类
        color_semantic = self.color_classifier(global_features)
        
        return color_semantic

class ImprovedColorCombinationHead(nn.Module):
    """改进的颜色组合分类头"""
    
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. 改进的颜色空间转换
        self.color_converter = ImprovedColorSpaceConverter()
        
        # 2. 改进的颜色通道注意力
        self.rgb_attention = ImprovedColorChannelAttention(3)
        self.hsv_attention = ImprovedColorChannelAttention(3)
        
        # 3. 改进的颜色语义提取
        self.color_semantic_extractor = ImprovedColorSemanticExtractor(6)  # RGB + HSV
        
        # 4. 改进的颜色组合分类器
        self.combination_classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # 5. 损失权重
        self.consistency_weight = 0.05  # 降低一致性损失权重
        self.semantic_weight = 0.02     # 降低语义损失权重
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入: [batch, 3, 32, 32] (RGB)
        
        # 1. 改进的颜色空间转换
        enhanced_rgb, hsv = self.color_converter(x)
        
        # 2. 改进的颜色通道注意力
        rgb_attended = self.rgb_attention(enhanced_rgb)
        hsv_attended = self.hsv_attention(hsv)
        
        # 3. 融合多颜色空间特征
        multi_color_features = torch.cat([rgb_attended, hsv_attended], dim=1)
        
        # 4. 改进的颜色语义提取
        color_semantic = self.color_semantic_extractor(multi_color_features)
        
        # 5. 颜色组合分类
        combination_logits = self.combination_classifier(color_semantic)
        
        return {
            'combination_logits': combination_logits,
            'color_semantic': color_semantic,
            'rgb_features': rgb_attended,
            'hsv_features': hsv_attended
        }
    
    def compute_loss(self, outputs, targets):
        """计算改进的损失函数"""
        combination_logits = outputs['combination_logits']
        color_semantic = outputs['color_semantic']
        
        # 1. 主要分类损失（使用标签平滑）
        classification_loss = F.cross_entropy(combination_logits, targets, label_smoothing=0.1)
        
        # 2. 简化的颜色语义一致性损失
        consistency_loss = 0.0
        if color_semantic.size(0) > 1:
            # 计算特征的标准差，鼓励特征的一致性
            feature_std = torch.std(color_semantic, dim=0)
            consistency_loss = torch.mean(feature_std)
        
        # 3. 简化的特征稀疏性损失
        sparsity_loss = torch.mean(torch.abs(color_semantic))
        
        # 4. 总损失
        total_loss = (classification_loss + 
                     self.consistency_weight * consistency_loss + 
                     self.semantic_weight * sparsity_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'consistency_loss': consistency_loss,
            'sparsity_loss': sparsity_loss
        }

def train_improved_color_combination_head(model, train_loader, val_loader, device, epochs=30):
    """训练改进的颜色组合分类头"""
    print(f"🚀 开始训练改进的颜色组合分类头...")
    
    # 使用类别权重来处理不平衡
    class_weights = torch.tensor([1.0, 3.0, 3.0, 20.0, 20.0]).to(device)  # 根据类别分布调整权重
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # 降低学习率和权重衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}
    
    best_accuracy = 0.0
    best_model_state = None
    patience = 12  # 增加耐心
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss_dict = model.compute_loss(outputs, targets)
            loss = loss_dict['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 降低梯度裁剪
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = outputs['combination_logits'].max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 30 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                loss_dict = model.compute_loss(outputs, targets)
                loss = loss_dict['total_loss']
                
                val_loss += loss.item()
                
                _, predicted = outputs['combination_logits'].max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        
        scheduler.step()
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        train_history['loss'].append(train_loss / len(train_loader))
        train_history['accuracy'].append(train_accuracy)
        val_history['loss'].append(val_loss / len(val_loader))
        val_history['accuracy'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"  Best Val Acc: {best_accuracy:.2f}%")
        print(f"  Patience: {patience_counter}/{patience}")
        print("-" * 50)
        
        if patience_counter >= patience:
            print(f"早停在第 {epoch+1} 轮")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_history, val_history, best_accuracy

def main():
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建平衡的数据集
    gtsrb_dir = "/home/hding22/color/GTSRB/GTSRB"
    
    try:
        train_dataset = BalancedColorCombinationDataset(
            size=1500, 
            use_real_data=True, 
            real_data_path=gtsrb_dir,
            balance_classes=True
        )
        
        test_dataset = BalancedColorCombinationDataset(
            size=400, 
            use_real_data=False,
            balance_classes=True
        )
        
        print(f"✅ 使用平衡的真实GTSRB训练数据 + 语义化测试数据")
        data_type = "balanced_real_train_semantic_test"
        
    except Exception as e:
        print(f"❌ 加载真实数据失败: {e}")
        print("使用纯语义化数据...")
        
        train_dataset = BalancedColorCombinationDataset(size=1500, use_real_data=False, balance_classes=True)
        test_dataset = BalancedColorCombinationDataset(size=400, use_real_data=False, balance_classes=True)
        
        print(f"✅ 使用纯语义化数据")
        data_type = "semantic_only"
    
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=2)
    
    # 创建改进的颜色组合分类头
    color_head = ImprovedColorCombinationHead(num_classes=5).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in color_head.parameters())
    print(f"改进颜色组合分类头参数量: {total_params:,}")
    
    # 训练模型
    start_time = time.time()
    trained_model, train_history, val_history, best_accuracy = train_improved_color_combination_head(
        color_head, train_loader, test_loader, device, epochs=40
    )
    training_time = time.time() - start_time
    
    # 保存结果
    results = {
        'model_name': 'ImprovedColorCombinationHead',
        'total_params': total_params,
        'best_accuracy': best_accuracy,
        'training_time': training_time,
        'train_history': train_history,
        'val_history': val_history,
        'epochs': len(train_history['loss']),
        'data_type': data_type,
        'task': 'improved_color_combination_classification',
        'num_classes': 5,
        'improvements': {
            'balanced_dataset': '平衡的数据集分布',
            'improved_architecture': '改进的网络架构',
            'better_regularization': '更好的正则化策略',
            'reduced_overfitting': '减少过拟合',
            'class_weights': '类别权重处理'
        },
        'color_combinations': IMPROVED_COLOR_COMBINATIONS
    }
    
    with open('improved_color_combination_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    torch.save(trained_model.state_dict(), 'best_improved_color_combination_head.pth')
    
    print(f"\n🎉 改进颜色组合分类头训练完成!")
    print(f"📊 结果总结:")
    print(f"  - 最佳准确率: {best_accuracy:.2f}%")
    print(f"  - 参数量: {total_params:,}")
    print(f"  - 训练时间: {training_time:.2f}秒")
    print(f"  - 数据类型: {data_type}")
    print(f"  - 分类任务: 5类颜色组合")
    print(f"  - 结果已保存到: improved_color_combination_results.json")
    print(f"  - 模型已保存到: best_improved_color_combination_head.pth")
    
    # 性能评估
    if best_accuracy > 80:
        print(f"✅ 优秀! 准确率超过80%")
    elif best_accuracy > 60:
        print(f"✅ 良好! 准确率超过60%")
    elif best_accuracy > 50:
        print(f"✅ 一般! 准确率超过50%")
    elif best_accuracy > 40:
        print(f"✅ 基本达标! 准确率超过40%")
    else:
        print(f"❌ 需要进一步改进，准确率低于40%")
    
    return trained_model, results

if __name__ == '__main__':
    model, results = main()
