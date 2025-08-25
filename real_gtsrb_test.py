# real_gtsrb_test.py
# 在真实GTSRB数据集上测试修复后的颜色头

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import sys
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/hding22/gtsrb_color_project/color_bias')

def download_gtsrb_data(data_root='/home/hding22/gtsrb_color_project/data'):
    """下载GTSRB数据集"""
    import urllib.request
    import zipfile
    
    gtsrb_dir = os.path.join(data_root, 'GTSRB')
    
    if os.path.exists(gtsrb_dir):
        print(f"GTSRB数据集已存在于 {gtsrb_dir}")
        return gtsrb_dir
    
    print("下载GTSRB数据集...")
    os.makedirs(data_root, exist_ok=True)
    
    # 下载URL
    url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    zip_path = os.path.join(data_root, "GTSRB_Final_Training_Images.zip")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("下载完成!")
    except Exception as e:
        print(f"下载失败: {e}")
        return None
    
    # 解压
    print("解压数据集...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_root)
    
    # 清理
    os.remove(zip_path)
    print("数据集准备完成!")
    return gtsrb_dir

class GTSRBDataset(torch.utils.data.Dataset):
    """简化的GTSRB数据集"""
    
    def __init__(self, data_root, split='train', img_size=32, transform=None):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((img_size + 4, img_size + 4)),
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(p=0.1),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        self.data, self.labels = self._load_data()
        
        # 分割数据
        if split in ['train', 'val']:
            train_data, val_data, train_labels, val_labels = train_test_split(
                self.data, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
            if split == 'train':
                self.data, self.labels = train_data, train_labels
            else:
                self.data, self.labels = val_data, val_labels
    
    def _load_data(self):
        """加载图像路径和标签"""
        images_dir = os.path.join(self.data_root, 'Final_Training', 'Images')
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录未找到: {images_dir}")
        
        data = []
        labels = []
        
        # 加载训练数据（类别目录）
        for class_id in range(43):
            class_dir = os.path.join(images_dir, f'{class_id:05d}')
            
            if os.path.exists(class_dir):
                # 读取CSV文件
                csv_file = os.path.join(class_dir, f'GT-{class_id:05d}.csv')
                
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, delimiter=';')
                    
                    for _, row in df.iterrows():
                        img_file = row['Filename']
                        img_path = os.path.join(class_dir, img_file)
                        
                        if os.path.exists(img_path):
                            data.append(img_path)
                            labels.append(class_id)
                else:
                    # 备用方案：加载所有.ppm文件
                    for file in os.listdir(class_dir):
                        if file.endswith('.ppm'):
                            img_path = os.path.join(class_dir, file)
                            data.append(img_path)
                            labels.append(class_id)
        
        print(f"加载了 {len(data)} 张图像，共 {len(set(labels))} 个类别")
        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        # 加载和转换图像
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label

def create_gtsrb_models():
    """创建GTSRB测试模型"""
    print("🔍 创建GTSRB测试模型...")
    
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
        
        def compute_loss(self, outputs, targets, images=None):
            """修复的损失函数"""
            color_logits = outputs['color_semantic_logits']
            loss = nn.CrossEntropyLoss()(color_logits, targets)
            
            return {
                'total_loss': loss,
                'main_loss': loss
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
        
        def compute_loss(self, outputs, targets, images=None):
            """集成损失函数"""
            final_logits = outputs['final_logits']
            loss = nn.CrossEntropyLoss()(final_logits, targets)
            
            return {
                'total_loss': loss,
                'main_loss': loss
            }
    
    fixed_color_head = FixedColorHead()
    base_model = BaseModel()
    ensemble_model = EnsembleModel()
    
    return fixed_color_head, base_model, ensemble_model

def train_gtsrb_model(model, train_loader, val_loader, model_name, device, epochs=20):
    """训练GTSRB模型"""
    print(f"🔍 开始训练 {model_name}...")
    
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}
    
    best_accuracy = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # 前向传播
            if hasattr(model, 'compute_loss'):
                if hasattr(model, 'color_head'):  # 集成模型
                    outputs = model(data)
                    loss_dict = model.compute_loss(outputs, targets, data)
                    loss = loss_dict['total_loss']
                    _, predicted = outputs['final_logits'].max(1)
                else:  # 颜色头
                    features = data.view(data.size(0), -1)[:, :64]
                    outputs = model(features)
                    loss_dict = model.compute_loss(outputs, targets, data)
                    loss = loss_dict['total_loss']
                    _, predicted = outputs['color_semantic_logits'].max(1)
            else:
                outputs = model(data)
                loss = criterion(outputs['logits'], targets)
                _, predicted = outputs['logits'].max(1)
            
            # 检查损失值
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                print(f"⚠️ 警告: 异常损失值 {loss.item()}")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                if hasattr(model, 'compute_loss'):
                    if hasattr(model, 'color_head'):  # 集成模型
                        outputs = model(data)
                        loss_dict = model.compute_loss(outputs, targets, data)
                        loss = loss_dict['total_loss']
                        _, predicted = outputs['final_logits'].max(1)
                    else:  # 颜色头
                        features = data.view(data.size(0), -1)[:, :64]
                        outputs = model(features)
                        loss_dict = model.compute_loss(outputs, targets, data)
                        loss = loss_dict['total_loss']
                        _, predicted = outputs['color_semantic_logits'].max(1)
                else:
                    outputs = model(data)
                    loss = criterion(outputs['logits'], targets)
                    _, predicted = outputs['logits'].max(1)
                
                val_loss += loss.item()
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # 计算准确率
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_history['loss'].append(train_loss / len(train_loader))
        train_history['accuracy'].append(train_accuracy)
        val_history['loss'].append(val_loss / len(val_loader))
        val_history['accuracy'].append(val_accuracy)
        
        print(f"  Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")
        
        # 早停机制
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f"best_gtsrb_{model_name.replace(' ', '_')}.pth")
            print(f"  🎉 新的最佳准确率: {best_accuracy:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️ 早停触发，最佳准确率: {best_accuracy:.2f}%")
                break
    
    return train_history, val_history, best_accuracy

def run_real_gtsrb_test():
    """运行真实GTSRB测试"""
    print("🔍 开始真实GTSRB测试")
    print("="*80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 下载/加载GTSRB数据集
    data_root = '/home/hding22/gtsrb_color_project/data'
    gtsrb_dir = download_gtsrb_data(data_root)
    
    if gtsrb_dir is None:
        print("❌ 无法获取GTSRB数据集")
        return
    
    # 创建数据集
    print("🔍 创建GTSRB数据集...")
    train_dataset = GTSRBDataset(gtsrb_dir, split='train', img_size=32)
    val_dataset = GTSRBDataset(gtsrb_dir, split='val', img_size=32)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"数据集大小: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")
    
    # 创建模型
    fixed_color_head, base_model, ensemble_model = create_gtsrb_models()
    fixed_color_head = fixed_color_head.to(device)
    base_model = base_model.to(device)
    ensemble_model = ensemble_model.to(device)
    
    results = {}
    
    # 实验1: 修复的颜色头
    print(f"\n🔬 实验1: 修复的颜色头")
    start_time = time.time()
    color_train_history, color_val_history, color_best_accuracy = train_gtsrb_model(
        fixed_color_head, train_loader, val_loader, "修复颜色头", device, epochs=20
    )
    color_training_time = time.time() - start_time
    
    results['fixed_color_head'] = {
        'best_accuracy': color_best_accuracy,
        'training_time': color_training_time,
        'train_history': color_train_history,
        'val_history': color_val_history
    }
    
    print(f"修复颜色头结果: {color_best_accuracy:.2f}% ({color_training_time:.2f}秒)")
    
    # 实验2: 基础模型
    print(f"\n🔬 实验2: 基础模型")
    start_time = time.time()
    base_train_history, base_val_history, base_best_accuracy = train_gtsrb_model(
        base_model, train_loader, val_loader, "基础模型", device, epochs=20
    )
    base_training_time = time.time() - start_time
    
    results['base_model'] = {
        'best_accuracy': base_best_accuracy,
        'training_time': base_training_time,
        'train_history': base_train_history,
        'val_history': base_val_history
    }
    
    print(f"基础模型结果: {base_best_accuracy:.2f}% ({base_training_time:.2f}秒)")
    
    # 实验3: 集成模型
    print(f"\n🔬 实验3: 集成模型")
    start_time = time.time()
    ensemble_train_history, ensemble_val_history, ensemble_best_accuracy = train_gtsrb_model(
        ensemble_model, train_loader, val_loader, "集成模型", device, epochs=20
    )
    ensemble_training_time = time.time() - start_time
    
    results['ensemble_model'] = {
        'best_accuracy': ensemble_best_accuracy,
        'training_time': ensemble_training_time,
        'train_history': ensemble_train_history,
        'val_history': ensemble_val_history
    }
    
    print(f"集成模型结果: {ensemble_best_accuracy:.2f}% ({ensemble_training_time:.2f}秒)")
    
    # 结果总结
    print(f"\n📊 真实GTSRB测试结果总结")
    print("="*60)
    
    print(f"模型对比:")
    print(f"  修复颜色头: {results['fixed_color_head']['best_accuracy']:.2f}%")
    print(f"  基础模型: {results['base_model']['best_accuracy']:.2f}%")
    print(f"  集成模型: {results['ensemble_model']['best_accuracy']:.2f}%")
    
    # 找出最佳模型
    best_model = max(results.keys(), key=lambda k: results[k]['best_accuracy'])
    best_accuracy = results[best_model]['best_accuracy']
    
    print(f"\n🏆 最佳模型: {best_model} ({best_accuracy:.2f}%)")
    
    # 性能分析
    print(f"\n🎯 性能分析:")
    print(f"  相对随机猜测: {best_accuracy/2.33:.1f}倍")
    print(f"  颜色头相对基础模型: {results['fixed_color_head']['best_accuracy'] - results['base_model']['best_accuracy']:.2f}%")
    
    if results['fixed_color_head']['best_accuracy'] > results['base_model']['best_accuracy']:
        print("✅ 颜色头表现更好！")
    else:
        print("⚠️ 基础模型表现更好")
    
    # 保存结果
    with open("real_gtsrb_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 结果已保存到: real_gtsrb_results.json")
    print("🎉 真实GTSRB测试完成!")
    
    return results

if __name__ == '__main__':
    run_real_gtsrb_test()

