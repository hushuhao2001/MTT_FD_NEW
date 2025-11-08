import torch
import sys
sys.path.append('/data/hushuhao-20251019/projects/MTT_FD')

# 加载数据
images = torch.load('/data/hushuhao-20251019/projects/MTT_FD/distilled_data/MTT/CIFAR100/IPC_3/images_best.pt')
labels = torch.load('/data/hushuhao-20251019/projects/MTT_FD/distilled_data/MTT/CIFAR100/IPC_3/labels_best.pt')

print(f"✅ 成功加载文件")
print(f"Images shape: {images.shape}")  # 应该是 [900, 3, 32, 32]
print(f"Labels shape: {labels.shape}")  # 应该是 [900]
print(f"Images dtype: {images.dtype}")
print(f"Labels dtype: {labels.dtype}")
print(f"\nImages统计:")
print(f"  Min: {images.min():.4f}")
print(f"  Max: {images.max():.4f}")
print(f"  Mean: {images.mean():.4f}")
print(f"  Std: {images.std():.4f}")
print(f"\nLabels统计:")
print(f"  Unique classes: {len(torch.unique(labels))}")  # 应该是100
print(f"  Images per class: {len(labels) // len(torch.unique(labels))}")  # 应该是9

# 检查每个类别的图像数量
for c in range(100):
    count = (labels == c).sum().item()
    if count != 9:
        print(f"⚠️ Warning: Class {c} has {count} images (expected 9)")

print("\n✅ 文件验证完成！")