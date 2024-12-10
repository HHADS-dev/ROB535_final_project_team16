import os
import torch

# 加载权重文件
pth_file = '/mnt/ws-frb/users/yiliuhh/mmpretraining/mmdetection3d/pretrained/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'
checkpoint = torch.load(pth_file, map_location='cpu')

# 统计参数量
total_params = sum(p.numel() for p in checkpoint['state_dict'].values())
print(f"模型参数量: {total_params / 1e6:.2f} M")

