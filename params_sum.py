import os
import torch

# 加载权重文件
pth_file1 = '/mnt/ws-frb/users/yiliuhh/mmpretraining/mmdetection3d/pretrained/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'
checkpoint1 = torch.load(pth_file1, map_location='cpu')

pth_file2 = '/mnt/ws-frb/users/yiliuhh/mmpretraining/mmdetection3d/pretrained/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
checkpoint2 = torch.load(pth_file2, map_location='cpu')

pth_file3 = '/mnt/ws-frb/users/yiliuhh/mmpretraining/VoxelNeXt/pretrained/voxelnext-pretrain.pth'
checkpoint3 = torch.load(pth_file3, map_location='cpu')
# 统计参数量
total_params1 = sum(p.numel() for p in checkpoint1['state_dict'].values())
print(f"模型参数量: {total_params1 / 1e6:.2f} M")

total_params2 = sum(p.numel() for p in checkpoint2['state_dict'].values())
print(f"模型参数量: {total_params2 / 1e6:.2f} M")
total_params3 = sum(p.numel() for p in checkpoint3.values())
print(f"模型参数量: {total_params3 / 1e6:.2f} M")
