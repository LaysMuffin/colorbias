import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from collections import defaultdict
import argparse
import random

# Import model
from improved_color_combination import (
    ImprovedColorCombinationHead, 
    BalancedColorCombinationDataset,
    IMPROVED_COLOR_COMBINATIONS
)

class RealGTSRBCAMGeneratorEnglish:
    """Real GTSRB Image CAM Generator with English Labels"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedColorCombinationHead(num_classes=5).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Get features from last convolutional layer
        self.features = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📱 Using device: {self.device}")
    
    def _register_hooks(self):
        """Register hooks to get features and gradients"""
        def forward_hook(module, input, output):
            self.features = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get last convolutional layer
        last_conv_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
        
        if last_conv_layer is not None:
            last_conv_layer.register_forward_hook(forward_hook)
            last_conv_layer.register_backward_hook(backward_hook)
            print(f"✅ Hooks registered to conv layer: {last_conv_layer}")
        else:
            print("❌ No conv layer found")
    
    def generate_cam(self, image, target_class=None):
        """Generate CAM"""
        # Prepare input
        if isinstance(image, np.ndarray):
            # Convert HWC to CHW format if needed
            if image.shape[-1] == 3:  # HWC -> CHW
                image = np.transpose(image, (2, 0, 1))
            image = torch.FloatTensor(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            # Convert HWC to CHW format if needed
            if image.shape[-1] == 3:  # HWC -> CHW
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad = True
        
        # Forward pass
        outputs = self.model(image)
        logits = outputs['combination_logits']
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Calculate score for target class
        score = logits[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get features and gradients
        if self.features is None or self.gradients is None:
            print("❌ Cannot get features or gradients")
            return None
        
        # Calculate weights
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # Generate CAM
        cam = torch.zeros(self.features.shape[2:], dtype=torch.float32, device=self.device)
        
        for i, w in enumerate(weights[0]):
            cam += w * self.features[0, i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, image, cam, save_path=None, alpha=0.6, title="CAM Visualization"):
        """Visualize CAM with English labels"""
        # Ensure image is numpy array
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu().numpy()
            if image.shape[0] == 3:  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
        
        # Resize CAM to match image
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Convert image format
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        else:
            image = np.uint8(image)
        
        # Overlay
        cam_image = heatmap + alpha * np.float32(image) / 255
        cam_image = cam_image / np.max(cam_image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # CAM heatmap
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay result
        axes[2].imshow(cam_image)
        axes[2].set_title('CAM Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ CAM image saved: {save_path}")
        
        plt.show()
        
        return cam_image

def load_gtsrb_samples_detailed_english(gtsrb_path, samples_per_combo=3):
    """Load detailed GTSRB samples with English labels"""
    samples = []
    
    # English color combination names
    english_names = {
        0: "Red-White-Black",
        1: "Blue-White", 
        2: "Yellow-White-Black",
        3: "Red-White",
        4: "Black-White"
    }
    
    if not os.path.exists(gtsrb_path):
        print(f"❌ GTSRB path does not exist: {gtsrb_path}")
        return samples
    
    # Load samples for each color combination
    for combo_id in range(5):
        combo = IMPROVED_COLOR_COMBINATIONS[combo_id]
        classes = combo['classes']
        
        print(f"📸 Loading samples for {english_names[combo_id]} combination...")
        
        # Load samples for each class
        for class_id in classes[:3]:  # Max 3 classes per combination
            class_folder = os.path.join(gtsrb_path, 'Final_Training', 'Images', f'{class_id:05d}')
            
            if os.path.exists(class_folder):
                # Get all images in this class
                image_files = [f for f in os.listdir(class_folder) if f.endswith('.ppm')]
                
                # Randomly select samples
                selected_files = random.sample(image_files, min(samples_per_combo, len(image_files)))
                
                for img_name in selected_files:
                    img_path = os.path.join(class_folder, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((32, 32))
                        img_array = np.array(img).astype(np.float32) / 255.0
                        
                        samples.append({
                            'image': img_array,
                            'label': combo_id,
                            'name': f"{english_names[combo_id]} (GTSRB {class_id})",
                            'class_id': class_id,
                            'combo_id': combo_id,
                            'file_name': img_name
                        })
                    except Exception as e:
                        print(f"❌ Failed to load image {img_path}: {e}")
                        continue
    
    return samples

def analyze_cam_results_english(cam_generator, samples, output_dir):
    """Analyze CAM results with English labels"""
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n🎯 Analyzing sample {i+1}/{len(samples)}: {sample['name']}")
        
        # Generate CAM
        cam = cam_generator.generate_cam(sample['image'], sample['label'])
        
        if cam is not None:
            # Save path
            save_path = os.path.join(output_dir, f'cam_{sample["name"]}_{sample["file_name"]}.png')
            
            # Visualize
            cam_generator.visualize_cam(sample['image'], cam, save_path, 
                                      title=f"{sample['name']} - {sample['file_name']}")
            
            # Save original CAM data
            cam_data_path = os.path.join(output_dir, f'cam_{sample["name"]}_{sample["file_name"]}.npy')
            np.save(cam_data_path, cam)
            
            # Analyze CAM
            cam_analysis = analyze_single_cam_english(cam, sample)
            results.append(cam_analysis)
            
            print(f"💾 CAM data saved: {cam_data_path}")
        else:
            print(f"❌ Cannot generate CAM for {sample['name']}")
    
    return results

def analyze_single_cam_english(cam, sample):
    """Analyze single CAM with English labels"""
    # Calculate CAM statistics
    cam_mean = np.mean(cam)
    cam_std = np.std(cam)
    cam_max = np.max(cam)
    cam_min = np.min(cam)
    
    # Calculate attention concentration (ratio of high-value regions)
    attention_threshold = 0.5
    high_attention_ratio = np.sum(cam > attention_threshold) / cam.size
    
    # Calculate spatial distribution
    center_x, center_y = cam.shape[1] // 2, cam.shape[0] // 2
    center_attention = np.mean(cam[center_y-4:center_y+4, center_x-4:center_x+4])
    
    return {
        'sample_name': sample['name'],
        'class_id': sample['class_id'],
        'combo_id': sample['combo_id'],
        'file_name': sample['file_name'],
        'cam_stats': {
            'mean': float(cam_mean),
            'std': float(cam_std),
            'max': float(cam_max),
            'min': float(cam_min),
            'high_attention_ratio': float(high_attention_ratio),
            'center_attention': float(center_attention)
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Generate Real GTSRB Image CAM Visualizations with English Labels')
    parser.add_argument('--model_path', default='best_improved_color_combination_head.pth', 
                       help='Model weight file path')
    parser.add_argument('--gtsrb_path', default='/home/hding22/color/GTSRB/GTSRB',
                       help='GTSRB dataset path')
    parser.add_argument('--output_dir', default='real_gtsrb_cam_english',
                       help='Output directory')
    parser.add_argument('--samples_per_combo', type=int, default=2,
                       help='Number of samples per color combination')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    random.seed(42)
    
    # Initialize CAM generator
    cam_generator = RealGTSRBCAMGeneratorEnglish(args.model_path)
    
    # Load real GTSRB samples
    print("📸 Loading real GTSRB samples...")
    samples = load_gtsrb_samples_detailed_english(args.gtsrb_path, args.samples_per_combo)
    
    print(f"✅ Ready to generate CAM visualizations for {len(samples)} real GTSRB samples")
    
    # Analyze CAM results
    results = analyze_cam_results_english(cam_generator, samples, args.output_dir)
    
    print(f"\n🎉 Real GTSRB CAM visualization with English labels completed! Results saved in: {args.output_dir}")
    
    # Generate detailed analysis report
    analysis_report = {
        'model_path': args.model_path,
        'output_dir': args.output_dir,
        'num_samples': len(samples),
        'data_type': 'Real GTSRB with English Labels',
        'samples_per_combo': args.samples_per_combo,
        'cam_analysis': results,
        'summary_stats': generate_summary_stats_english(results)
    }
    
    # Save analysis report
    report_path = os.path.join(args.output_dir, 'detailed_cam_analysis_english.json')
    with open(report_path, 'w') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Detailed analysis report saved: {report_path}")
    
    # Generate visualization summary
    generate_visualization_summary_english(results, args.output_dir)

def generate_summary_stats_english(results):
    """Generate summary statistics with English labels"""
    if not results:
        return {}
    
    # English color combination names
    english_names = {
        0: "Red-White-Black",
        1: "Blue-White", 
        2: "Yellow-White-Black",
        3: "Red-White",
        4: "Black-White"
    }
    
    # Group by color combination
    combo_stats = defaultdict(list)
    for result in results:
        combo_id = result['combo_id']
        combo_stats[combo_id].append(result['cam_stats'])
    
    summary = {}
    for combo_id, stats_list in combo_stats.items():
        combo_name = english_names[combo_id]
        summary[combo_name] = {
            'num_samples': len(stats_list),
            'avg_attention_ratio': np.mean([s['high_attention_ratio'] for s in stats_list]),
            'avg_center_attention': np.mean([s['center_attention'] for s in stats_list]),
            'avg_cam_mean': np.mean([s['mean'] for s in stats_list]),
            'avg_cam_std': np.mean([s['std'] for s in stats_list])
        }
    
    return summary

def generate_visualization_summary_english(results, output_dir):
    """Generate visualization summary with English labels"""
    if not results:
        return
    
    # English color combination names
    english_names = {
        0: "Red-White-Black",
        1: "Blue-White", 
        2: "Yellow-White-Black",
        3: "Red-White",
        4: "Black-White"
    }
    
    # Create summary charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Attention ratio by color combination
    combo_attention = defaultdict(list)
    for result in results:
        combo_name = english_names[result['combo_id']]
        combo_attention[combo_name].append(result['cam_stats']['high_attention_ratio'])
    
    # Plot attention ratio boxplot
    attention_data = [combo_attention[name] for name in combo_attention.keys()]
    axes[0, 0].boxplot(attention_data, tick_labels=list(combo_attention.keys()))
    axes[0, 0].set_title('Attention Ratio by Color Combination', fontweight='bold')
    axes[0, 0].set_ylabel('High Attention Ratio')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot center attention distribution
    center_attention = [r['cam_stats']['center_attention'] for r in results]
    axes[0, 1].hist(center_attention, bins=20, alpha=0.7, color='blue')
    axes[0, 1].set_title('Center Attention Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Center Attention')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot CAM mean distribution
    cam_means = [r['cam_stats']['mean'] for r in results]
    axes[1, 0].hist(cam_means, bins=20, alpha=0.7, color='green')
    axes[1, 0].set_title('CAM Mean Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('CAM Mean')
    axes[1, 0].set_ylabel('Frequency')
    
    # Plot CAM standard deviation distribution
    cam_stds = [r['cam_stats']['std'] for r in results]
    axes[1, 1].hist(cam_stds, bins=20, alpha=0.7, color='red')
    axes[1, 1].set_title('CAM Standard Deviation Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('CAM Standard Deviation')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save summary chart
    summary_path = os.path.join(output_dir, 'cam_analysis_summary_english.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"📊 Analysis summary chart saved: {summary_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
