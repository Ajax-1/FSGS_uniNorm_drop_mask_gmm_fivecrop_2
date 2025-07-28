import os
import numpy as np
import cv2
import torch

def depth_vis(save_path, filename, depth_tensor):
    if not os.path.exists(save_path):
            os.makedirs(save_path) 

    # 保存midas深度
    if isinstance(depth_tensor, torch.Tensor):
        midas_depth = depth_tensor.detach().cpu().numpy()                
    else:
        midas_depth = depth_tensor

    # 归一化
    midas_min = np.min(midas_depth)
    midas_max = np.max(midas_depth)
    if midas_max > midas_min:  # 避免除以0
        midas_normalized = ((midas_depth - midas_min) / (midas_max - midas_min) * 255).astype(np.uint8)
    else:
        midas_normalized = np.zeros_like(midas_depth, dtype=np.uint8)
    
    cv2.imwrite('{}/{}.jpg'.format(save_path, filename), midas_normalized)