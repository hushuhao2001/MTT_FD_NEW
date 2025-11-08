import os
import argparse
import torch
import numpy as np


def subsample_dataset(input_img_path, input_label_path, output_img_path, output_label_path, 
                      input_ipc, output_ipc, num_classes=100, seed=42):
    """
    从IPC较大的数据集中随机采样生成IPC较小的数据集
    
    Args:
        input_img_path: 输入的images_best.pt路径
        input_label_path: 输入的labels_best.pt路径
        output_img_path: 输出的images_best.pt路径
        output_label_path: 输出的labels_best.pt路径
        input_ipc: 输入数据的IPC（每类图像数）
        output_ipc: 输出数据的IPC（每类图像数）
        num_classes: 类别数（CIFAR100默认100）
        seed: 随机种子
    """
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("="*80)
    print("Dataset Subsampling Tool")
    print("="*80)
    print(f"Input IPC:  {input_ipc}")
    print(f"Output IPC: {output_ipc}")
    print(f"Classes:    {num_classes}")
    print(f"Seed:       {seed}")
    print("="*80)
    
    # 验证参数
    assert output_ipc <= input_ipc, f"Output IPC ({output_ipc}) must be <= Input IPC ({input_ipc})"
    
    # 加载数据
    print(f"\n📂 Loading data from:")
    print(f"   Images: {input_img_path}")
    print(f"   Labels: {input_label_path}")
    
    images = torch.load(input_img_path)
    labels = torch.load(input_label_path)
    
    print(f"\n✅ Loaded successfully!")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # 验证数据
    expected_total = num_classes * input_ipc
    assert images.shape[0] == expected_total, f"Expected {expected_total} images, got {images.shape[0]}"
    assert labels.shape[0] == expected_total, f"Expected {expected_total} labels, got {labels.shape[0]}"
    assert len(torch.unique(labels)) == num_classes, f"Expected {num_classes} classes, got {len(torch.unique(labels))}"
    
    print(f"\n✅ Data validation passed!")
    
    # 按类别组织数据
    print(f"\n🔄 Subsampling from IPC={input_ipc} to IPC={output_ipc}...")
    
    selected_images = []
    selected_labels = []
    
    for c in range(num_classes):
        # 找到该类的所有样本索引
        class_indices = torch.where(labels == c)[0]
        
        # 验证该类有足够的样本
        assert len(class_indices) == input_ipc, f"Class {c} has {len(class_indices)} samples, expected {input_ipc}"
        
        # 随机选择output_ipc个样本
        selected_idx = np.random.choice(class_indices.numpy(), size=output_ipc, replace=False)
        
        # 收集选中的图像和标签
        selected_images.append(images[selected_idx])
        selected_labels.append(labels[selected_idx])
        
        if (c + 1) % 20 == 0:
            print(f"   Processed {c+1}/{num_classes} classes...")
    
    # 拼接所有类别
    output_images = torch.cat(selected_images, dim=0)
    output_labels = torch.cat(selected_labels, dim=0)
    
    print(f"\n✅ Subsampling completed!")
    print(f"   Output images shape: {output_images.shape}")
    print(f"   Output labels shape: {output_labels.shape}")
    
    # 验证输出数据
    assert output_images.shape[0] == num_classes * output_ipc
    assert output_labels.shape[0] == num_classes * output_ipc
    
    # 验证每个类别的样本数
    for c in range(num_classes):
        count = (output_labels == c).sum().item()
        assert count == output_ipc, f"Class {c} has {count} samples, expected {output_ipc}"
    
    print(f"\n✅ Output data validation passed!")
    
    # 创建输出目录
    output_img_dir = os.path.dirname(output_img_path)
    output_label_dir = os.path.dirname(output_label_path)
    
    if output_img_dir and not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        print(f"\n📁 Created directory: {output_img_dir}")
    
    if output_label_dir and output_label_dir != output_img_dir and not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
        print(f"📁 Created directory: {output_label_dir}")
    
    # 保存数据
    print(f"\n💾 Saving data to:")
    print(f"   Images: {output_img_path}")
    print(f"   Labels: {output_label_path}")
    
    torch.save(output_images, output_img_path)
    torch.save(output_labels, output_label_path)
    
    print(f"\n✅ Saved successfully!")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("📊 Summary:")
    print("="*80)
    print(f"Input:  {num_classes} classes × {input_ipc} images = {num_classes * input_ipc} total")
    print(f"Output: {num_classes} classes × {output_ipc} images = {num_classes * output_ipc} total")
    print(f"Reduction: {(1 - output_ipc/input_ipc)*100:.1f}%")
    print("="*80)
    print("✅ All done!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subsample dataset from higher IPC to lower IPC')
    
    parser.add_argument('--input-img', type=str, required=True, 
                        help='Path to input images_best.pt')
    parser.add_argument('--input-label', type=str, required=True,
                        help='Path to input labels_best.pt')
    parser.add_argument('--output-img', type=str, required=True,
                        help='Path to output images_best.pt')
    parser.add_argument('--output-label', type=str, required=True,
                        help='Path to output labels_best.pt')
    parser.add_argument('--input-ipc', type=int, required=True,
                        help='Input IPC (images per class)')
    parser.add_argument('--output-ipc', type=int, required=True,
                        help='Output IPC (images per class, must be <= input-ipc)')
    parser.add_argument('--num-classes', type=int, default=100,
                        help='Number of classes (default: 100 for CIFAR100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    subsample_dataset(
        input_img_path=args.input_img,
        input_label_path=args.input_label,
        output_img_path=args.output_img,
        output_label_path=args.output_label,
        input_ipc=args.input_ipc,
        output_ipc=args.output_ipc,
        num_classes=args.num_classes,
        seed=args.seed
    )