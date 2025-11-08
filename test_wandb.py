# test_wandb.py
import argparse

class Args:
    def __init__(self):
        self.ipc = 9
        self.batch_syn = None
        self.lr_img = 1000
        self.dsa_param = "some_object"

args = Args()
print("Before wandb:")
print(f"  ipc = {args.ipc}")
print(f"  batch_syn = {args.batch_syn}")
print(f"  lr_img = {args.lr_img}")
print(f"  dsa_param = {args.dsa_param}")

# 模拟wandb处理
class WandbConfig:
    def __init__(self, args):
        self._items = {}
        # 只存储基本类型
        for key in ['ipc', 'batch_syn', 'lr_img']:
            if hasattr(args, key):
                val = getattr(args, key)
                if val is not None:  # wandb可能跳过None
                    self._items[key] = val

wandb_config = WandbConfig(args)
print("\nWandb config:")
print(f"  _items = {wandb_config._items}")

# 模拟清空和恢复
args = type('', (), {})()
for key in wandb_config._items:
    setattr(args, key, wandb_config._items[key])

print("\nAfter wandb restore:")
print(f"  ipc = {args.ipc if hasattr(args, 'ipc') else 'MISSING'}")
print(f"  batch_syn = {args.batch_syn if hasattr(args, 'batch_syn') else 'MISSING'}")
print(f"  lr_img = {args.lr_img if hasattr(args, 'lr_img') else 'MISSING'}")
print(f"  dsa_param = {args.dsa_param if hasattr(args, 'dsa_param') else 'MISSING'}")