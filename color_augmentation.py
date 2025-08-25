# color_augmentation.py
# 颜色特定的数据增强模块

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import random
from typing import Tuple, Dict
import math

class AdvancedColorSpecificAugmentation:
    """高级颜色特定数据增强 - 保持语义的同时增强颜色鲁棒性"""
    
    def __init__(self, 
                 brightness_range=(0.6, 1.4),
                 contrast_range=(0.7, 1.3),
                 saturation_range=(0.5, 1.5),
                 hue_range=(-0.15, 0.15),
                 gamma_range=(0.6, 1.4),
                 color_shift_prob=0.6,
                 lighting_prob=0.4,
                 semantic_preserve_prob=0.7):
        
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.gamma_range = gamma_range
        self.color_shift_prob = color_shift_prob
        self.lighting_prob = lighting_prob
        self.semantic_preserve_prob = semantic_preserve_prob
        
        # 标准颜色增强
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness_range[1] - 1.0,
            contrast=contrast_range[1] - 1.0,
            saturation=saturation_range[1] - 1.0,
            hue=hue_range[1]
        )
        
        # 颜色不变性增强器
        self.color_invariance_augmentor = ColorInvarianceAugmentor()
        
        # 语义保持增强器
        self.semantic_preserve_augmentor = SemanticPreserveAugmentor()
        
        # 对抗性颜色增强器
        self.adversarial_color_augmentor = AdversarialColorAugmentor()
    
    def __call__(self, image: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """应用高级颜色增强"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # 批次维度
                augmented_images = []
                for i in range(image.size(0)):
                    target = targets[i] if targets is not None else None
                    img = self._augment_single_image(image[i], target)
                    augmented_images.append(img)
                return torch.stack(augmented_images)
            else:
                return self._augment_single_image(image, targets)
        else:
            return self._augment_single_image(image, targets)
    
    def _augment_single_image(self, image: torch.Tensor, target: int = None) -> torch.Tensor:
        """增强单个图像"""
        # 转换为PIL图像
        if image.dim() == 3:
            img_array = image.permute(1, 2, 0).cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
        else:
            pil_image = image
        
        # 1. 基础颜色增强
        if random.random() < self.color_shift_prob:
            pil_image = self.color_jitter(pil_image)
        
        # 2. 光照变化
        if random.random() < self.lighting_prob:
            pil_image = self._adjust_lighting(pil_image)
        
        # 3. 形状保持的颜色增强
        if random.random() < 0.4:
            pil_image = self._shape_preserving_color_augmentation(pil_image)
        
        # 4. 颜色不变性增强
        if random.random() < 0.5:
            pil_image = self.color_invariance_augmentor(pil_image)
        
        # 5. 语义保持增强
        if random.random() < self.semantic_preserve_prob and target is not None:
            pil_image = self.semantic_preserve_augmentor(pil_image, target)
        
        # 6. 对抗性颜色增强
        if random.random() < 0.3:
            pil_image = self.adversarial_color_augmentor(pil_image)
        
        # 7. 自适应颜色增强
        if random.random() < 0.4:
            pil_image = self._adaptive_color_augmentation(pil_image, target)
        
        # 转换回tensor
        if isinstance(pil_image, Image.Image):
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        else:
            img_tensor = pil_image
        
        return img_tensor
    
    def _adjust_lighting(self, image: Image.Image) -> Image.Image:
        """调整光照条件"""
        # 随机gamma调整
        gamma = random.uniform(*self.gamma_range)
        return transforms.functional.adjust_gamma(image, gamma)
    
    def _shape_preserving_color_augmentation(self, image: Image.Image) -> Image.Image:
        """保持形状不变的颜色增强"""
        # 转换为HSV空间
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 只调整色调和饱和度，保持亮度（形状信息）
        hsv[:, :, 0] = hsv[:, :, 0] * random.uniform(0.8, 1.2)  # 色调
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.7, 1.3)  # 饱和度
        
        # 确保值在有效范围内
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)  # OpenCV HSV: H[0,179]
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # 转换回RGB
        hsv = hsv.astype(np.uint8)
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(augmented)
    
    def _adaptive_color_augmentation(self, image: Image.Image, target: int = None) -> Image.Image:
        """自适应颜色增强 - 根据类别调整增强策略"""
        if target is None:
            return image
        
        # 根据类别选择不同的增强策略
        if target in [0, 1, 2, 3, 4, 5, 7, 8]:  # 速度限制标志
            # 红色边框标志，增强红色通道
            return self._enhance_red_channel(image)
        elif target in [14, 17]:  # 停止和禁止标志
            # 红色填充标志，增强红色对比度
            return self._enhance_red_contrast(image)
        elif target in [33, 34, 35, 36, 37, 38, 39, 40]:  # 蓝色方向标志
            # 蓝色标志，增强蓝色通道
            return self._enhance_blue_channel(image)
        else:
            # 其他标志，通用增强
            return self._general_color_enhancement(image)
    
    def _enhance_red_channel(self, image: Image.Image) -> Image.Image:
        """增强红色通道"""
        img_array = np.array(image).astype(np.float32)
        
        # 增强红色通道
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.2, 0, 255)
        
        # 轻微抑制其他通道
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.9, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.9, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _enhance_red_contrast(self, image: Image.Image) -> Image.Image:
        """增强红色对比度"""
        img_array = np.array(image).astype(np.float32)
        
        # 增强红色对比度
        red_channel = img_array[:, :, 0]
        red_mean = np.mean(red_channel)
        red_std = np.std(red_channel)
        
        # 标准化并增强对比度
        red_normalized = (red_channel - red_mean) / (red_std + 1e-8)
        red_enhanced = red_normalized * 1.5 + red_mean
        
        img_array[:, :, 0] = np.clip(red_enhanced, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _enhance_blue_channel(self, image: Image.Image) -> Image.Image:
        """增强蓝色通道"""
        img_array = np.array(image).astype(np.float32)
        
        # 增强蓝色通道
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.2, 0, 255)
        
        # 轻微抑制其他通道
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.9, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.9, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _general_color_enhancement(self, image: Image.Image) -> Image.Image:
        """通用颜色增强"""
        img_array = np.array(image).astype(np.float32)
        
        # 增强整体饱和度
        hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
        
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(enhanced)

class ColorInvarianceAugmentor:
    """颜色不变性增强器 - 生成颜色不变的特征"""
    
    def __init__(self):
        self.lighting_variations = [
            (0.8, 1.2),  # 亮度变化
            (0.7, 1.3),  # 对比度变化
            (0.6, 1.4),  # 饱和度变化
        ]
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """应用颜色不变性增强"""
        # 随机选择光照变化
        variation = random.choice(self.lighting_variations)
        
        # 应用变化
        if random.random() < 0.5:
            # 亮度变化
            factor = random.uniform(*variation)
            return transforms.functional.adjust_brightness(image, factor)
        else:
            # 对比度变化
            factor = random.uniform(*variation)
            return transforms.functional.adjust_contrast(image, factor)

class SemanticPreserveAugmentor:
    """语义保持增强器 - 保持颜色语义的同时增强鲁棒性"""
    
    def __init__(self):
        # GTSRB颜色语义映射
        self.color_semantic_map = {
            'red': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
            'blue': [33, 34, 35, 36, 37, 38, 39, 40],
            'yellow': [12],
            'white': list(range(43)),  # 所有类别都有白色
            'black': [6, 9, 10, 11, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 41, 42]
        }
    
    def __call__(self, image: Image.Image, target: int) -> Image.Image:
        """应用语义保持增强"""
        # 确定目标类别的主要颜色
        primary_colors = self._get_primary_colors(target)
        
        # 根据主要颜色进行增强
        if 'red' in primary_colors:
            return self._preserve_red_semantics(image)
        elif 'blue' in primary_colors:
            return self._preserve_blue_semantics(image)
        elif 'yellow' in primary_colors:
            return self._preserve_yellow_semantics(image)
        else:
            return self._preserve_general_semantics(image)
    
    def _get_primary_colors(self, target: int) -> list:
        """获取目标类别的主要颜色"""
        primary_colors = []
        for color, classes in self.color_semantic_map.items():
            if target in classes:
                primary_colors.append(color)
        return primary_colors
    
    def _preserve_red_semantics(self, image: Image.Image) -> Image.Image:
        """保持红色语义"""
        img_array = np.array(image).astype(np.float32)
        
        # 保持红色通道的相对强度
        red_ratio = img_array[:, :, 0] / (img_array[:, :, 1] + img_array[:, :, 2] + 1e-8)
        
        # 增强红色语义
        red_enhanced = np.clip(red_ratio * 1.1, 0, 1)
        
        # 应用增强
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * red_enhanced, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _preserve_blue_semantics(self, image: Image.Image) -> Image.Image:
        """保持蓝色语义"""
        img_array = np.array(image).astype(np.float32)
        
        # 保持蓝色通道的相对强度
        blue_ratio = img_array[:, :, 2] / (img_array[:, :, 0] + img_array[:, :, 1] + 1e-8)
        
        # 增强蓝色语义
        blue_enhanced = np.clip(blue_ratio * 1.1, 0, 1)
        
        # 应用增强
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * blue_enhanced, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _preserve_yellow_semantics(self, image: Image.Image) -> Image.Image:
        """保持黄色语义"""
        img_array = np.array(image).astype(np.float32)
        
        # 黄色 = 红色 + 绿色
        yellow_intensity = (img_array[:, :, 0] + img_array[:, :, 1]) / 2
        
        # 增强黄色语义
        yellow_enhanced = np.clip(yellow_intensity * 1.1, 0, 255)
        
        # 应用增强
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.05, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.05, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _preserve_general_semantics(self, image: Image.Image) -> Image.Image:
        """保持通用语义"""
        # 轻微的颜色增强，保持整体语义
        return transforms.functional.adjust_saturation(image, 1.1)

class AdversarialColorAugmentor:
    """对抗性颜色增强器 - 生成具有挑战性的颜色变化"""
    
    def __init__(self):
        self.adversarial_strength = 0.3
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """应用对抗性颜色增强"""
        img_array = np.array(image).astype(np.float32)
        
        # 随机选择对抗性策略
        strategy = random.choice(['color_shift', 'channel_attack', 'semantic_confusion'])
        
        if strategy == 'color_shift':
            return self._color_shift_attack(img_array)
        elif strategy == 'channel_attack':
            return self._channel_attack(img_array)
        else:
            return self._semantic_confusion_attack(img_array)
    
    def _color_shift_attack(self, img_array: np.ndarray) -> Image.Image:
        """颜色偏移攻击"""
        # 随机颜色偏移
        shift = np.random.uniform(-30, 30, 3)
        
        for i in range(3):
            img_array[:, :, i] = np.clip(img_array[:, :, i] + shift[i], 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _channel_attack(self, img_array: np.ndarray) -> Image.Image:
        """通道攻击"""
        # 随机交换或抑制通道
        channels = [0, 1, 2]
        random.shuffle(channels)
        
        # 轻微抑制某些通道
        for i, channel in enumerate(channels):
            if random.random() < 0.5:
                img_array[:, :, channel] = img_array[:, :, channel] * 0.8
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _semantic_confusion_attack(self, img_array: np.ndarray) -> Image.Image:
        """语义混淆攻击"""
        # 转换为HSV并添加噪声
        hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 在色调通道添加噪声
        noise = np.random.normal(0, 10, hsv.shape[:2])
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + noise, 0, 179)
        
        # 转换回RGB
        confused = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(confused)

class ColorConsistencyLoss(nn.Module):
    """颜色一致性损失"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features_orig: torch.Tensor, features_aug: torch.Tensor) -> torch.Tensor:
        """计算颜色一致性损失"""
        # 归一化特征
        import torch.nn.functional as F
        features_orig_norm = F.normalize(features_orig, dim=1)
        features_aug_norm = F.normalize(features_aug, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(features_orig_norm, features_aug_norm.t()) / self.temperature
        
        # 对角线元素应该是最大的（同一图像的不同增强版本）
        labels = torch.arange(features_orig.size(0), device=features_orig.device)
        
        # 对比学习损失
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class ColorRobustnessLoss(nn.Module):
    """颜色鲁棒性损失"""
    
    def __init__(self, lambda_consistency=0.1, lambda_invariance=0.05):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.lambda_invariance = lambda_invariance
        self.consistency_loss = ColorConsistencyLoss()
    
    def forward(self, 
                features_orig: torch.Tensor, 
                features_aug: torch.Tensor,
                predictions_orig: torch.Tensor,
                predictions_aug: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算颜色鲁棒性损失"""
        
        # 一致性损失
        consistency_loss = self.consistency_loss(features_orig, features_aug)
        
        # 预测不变性损失
        import torch.nn.functional as F
        # 确保数据类型一致
        predictions_orig = predictions_orig.float()
        predictions_aug = predictions_aug.float()
        invariance_loss = F.mse_loss(predictions_orig, predictions_aug)
        
        # 总损失
        total_loss = (
            self.lambda_consistency * consistency_loss +
            self.lambda_invariance * invariance_loss
        )
        
        return {
            'total_loss': total_loss,
            'consistency_loss': consistency_loss,
            'invariance_loss': invariance_loss
        }

def test_color_augmentation():
    """测试高级颜色增强模块"""
    print("🎨 测试高级颜色增强模块")
    print("="*60)
    
    # 创建测试图像
    batch_size = 4
    channels = 3
    height, width = 32, 32
    
    # 创建彩色测试图像
    test_images = torch.randn(batch_size, channels, height, width)
    test_targets = torch.randint(0, 43, (batch_size,))
    
    print(f"测试图像形状: {test_images.shape}")
    print(f"测试目标形状: {test_targets.shape}")
    
    # 创建高级颜色增强器
    advanced_color_aug = AdvancedColorSpecificAugmentation()
    
    # 应用增强
    augmented_images = advanced_color_aug(test_images, test_targets)
    
    print(f"增强后图像形状: {augmented_images.shape}")
    
    # 测试颜色不变性增强器
    invariance_augmentor = ColorInvarianceAugmentor()
    test_image = torch.randn(channels, height, width)
    test_pil = transforms.ToPILImage()(test_image)
    invariance_augmented = invariance_augmentor(test_pil)
    print(f"颜色不变性增强: {type(invariance_augmented)}")
    
    # 测试语义保持增强器
    semantic_augmentor = SemanticPreserveAugmentor()
    semantic_augmented = semantic_augmentor(test_pil, 0)  # 速度限制标志
    print(f"语义保持增强: {type(semantic_augmented)}")
    
    # 测试对抗性颜色增强器
    adversarial_augmentor = AdversarialColorAugmentor()
    adversarial_augmented = adversarial_augmentor(test_pil)
    print(f"对抗性颜色增强: {type(adversarial_augmented)}")
    
    # 测试颜色一致性损失
    consistency_loss = ColorConsistencyLoss()
    features_orig = torch.randn(batch_size, 64)
    features_aug = torch.randn(batch_size, 64)
    
    loss = consistency_loss(features_orig, features_aug)
    print(f"颜色一致性损失: {loss.item():.4f}")
    
    # 测试颜色鲁棒性损失
    robustness_loss = ColorRobustnessLoss()
    predictions_orig = torch.randn(batch_size, 43)
    predictions_aug = torch.randn(batch_size, 43)
    
    loss_dict = robustness_loss(features_orig, features_aug, predictions_orig, predictions_aug)
    
    print(f"颜色鲁棒性损失:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print(f"\n✅ 高级颜色增强模块测试完成")

if __name__ == '__main__':
    test_color_augmentation()
