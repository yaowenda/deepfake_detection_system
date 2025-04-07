import torch

# 加载权重文件
checkpoint_path = '../checkpoints/M2TR_CelebDF_epoch_00019.pyth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 打印权重文件的基本信息
print("权重文件内容:")
if isinstance(checkpoint, dict):
    print("\n字典的键:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"- {key}: Tensor形状 {checkpoint[key].shape}")
        else:
            print(f"- {key}: {type(checkpoint[key])}")
else:
    print(f"权重文件类型: {type(checkpoint)}")
    if isinstance(checkpoint, torch.nn.Module):
        print("这是一个完整的模型")