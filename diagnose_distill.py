import argparse
import torch

def diagnose():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR100')
    parser.add_argument('--ipc', type=int, default=9)
    parser.add_argument('--batch_syn', type=int, default=None)
    parser.add_argument('--lr_img', type=float, default=1000)
    
    # 模拟你的命令行
    args = parser.parse_args(['--dataset', 'CIFAR100', '--ipc', '9'])
    
    num_classes = 100
    
    print("="*80)
    print("STEP 1: 原始参数")
    print("="*80)
    print(f"dataset    = {args.dataset}")
    print(f"ipc        = {args.ipc}")
    print(f"batch_syn  = {args.batch_syn}")
    print(f"lr_img     = {args.lr_img}")
    
    print("\n" + "="*80)
    print("STEP 2: 模拟wandb清空args")
    print("="*80)
    
    # 模拟wandb.config
    class WandbConfig:
        def __init__(self, original_args):
            self._items = {}
            for key, val in vars(original_args).items():
                # Wandb只保存非None的基本类型
                if val is not None and isinstance(val, (int, float, str, bool)):
                    self._items[key] = val
    
    wandb_config = WandbConfig(args)
    print(f"wandb.config._items = {wandb_config._items}")
    
    # 清空args
    args = type('', (), {})()
    
    # 从wandb恢复
    for key in wandb_config._items:
        setattr(args, key, wandb_config._items[key])
    
    print("\n" + "="*80)
    print("STEP 3: wandb恢复后的参数")
    print("="*80)
    print(f"dataset    = {getattr(args, 'dataset', 'MISSING')}")
    print(f"ipc        = {getattr(args, 'ipc', 'MISSING')}")
    print(f"batch_syn  = {getattr(args, 'batch_syn', 'MISSING')}")
    print(f"lr_img     = {getattr(args, 'lr_img', 'MISSING')}")
    
    print("\n" + "="*80)
    print("STEP 4: 尝试设置batch_syn")
    print("="*80)
    
    # 模拟第77行的代码
    try:
        if args.batch_syn is None:
            args.batch_syn = num_classes * args.ipc
            print(f"✅ batch_syn 被设置为: {args.batch_syn}")
        else:
            print(f"batch_syn 已有值: {args.batch_syn}")
    except AttributeError as e:
        print(f"❌ 错误: {e}")
        print("batch_syn 属性不存在！")
        # 强制设置
        if not hasattr(args, 'batch_syn'):
            args.batch_syn = num_classes * args.ipc
            print(f"强制设置 batch_syn = {args.batch_syn}")
    
    print("\n" + "="*80)
    print("STEP 5: 最终参数")
    print("="*80)
    print(f"batch_syn = {getattr(args, 'batch_syn', 'STILL MISSING')}")
    
    if hasattr(args, 'batch_syn'):
        if args.batch_syn == 900:
            print(f"\n⚠️⚠️⚠️ 警告: batch_syn = 900 太大了！")
            print(f"总共只有 {num_classes * 9} 张图，batch_syn应该 <= 256")
        elif args.batch_syn is None:
            print(f"\n❌ 错误: batch_syn 仍然是 None")
        else:
            print(f"\n✅ batch_syn = {args.batch_syn} 看起来合理")

if __name__ == '__main__':
    diagnose()