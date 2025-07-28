import numpy as np
import torch
from PIL import Image
from unidepth.models import UniDepthV2
from unidepth.utils import colorize
from transformers import AutoConfig
from pyntcloud import PyntCloud
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


local_model_path = "./lpiccinelli-eth/UniDepth/checkpoint/unidepth-v2-vitl14"
config = AutoConfig.from_pretrained(local_model_path).to_dict()
model = UniDepthV2.from_pretrained(
    local_model_path,
    config=config,
    local_files_only=True  # 强制不联网
)
model.interpolation_mode = "bilinear"  # 设置插值方式
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
downsampling = 1

def estimate_depth_uni(image_tensor: torch.Tensor):
    #import pdb;pdb.set_trace()    
    rgb_torch = image_tensor.float().to(model.device)
    h, w = rgb_torch.shape[1:3]
    norm_img = (rgb_torch[None] - 0.5) / 0.5
    '''
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)
    '''
    # 推理（无需相机内参的简化版）
    with torch.no_grad():
        #predictions = model.infer(rgb_torch)
        predictions = model.infer(norm_img)
    '''
    depth = torch.nn.functional.interpolate(
        depth,
        size=(h//downsampling, w//downsampling),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    '''
    #depth.unsqueeze(1),
    # 提取结果
    # depth = predictions["depth"].squeeze().cpu().numpy()  # 深度图 (H, W)
    # xyz = predictions["points"].cpu().numpy()             # 点云 (3, H, W)
    # intrinsics = predictions["intrinsics"].cpu().numpy()  # 内参 (3, 3)
    depth = predictions["depth"].squeeze()
    conf = predictions["confidence"].squeeze()
    
    import pdb;pdb.set_trace()
    #################################
    # 可视化并保存结果
    # 转换为numpy数组（移至CPU）
    depth_np = depth.cpu().numpy()
    conf_np = conf.cpu().numpy()
    
    # 创建颜色映射（深度图用蓝-绿-红，置信度用jet）
    depth_cmap = LinearSegmentedColormap.from_list('depth_cmap', ['blue', 'green', 'red'])
    
    # 创建图形并设置子图
    plt.figure(figsize=(12, 6))  # 调整尺寸更紧凑
    
    # 绘制深度图
    plt.subplot(121)
    depth_im = plt.imshow(depth_np, cmap=depth_cmap)
    plt.title('Depth Map')  # 英文标题更通用，也可改为中文
    plt.colorbar(depth_im, label='Depth Value')  # 颜色条标注
    plt.axis('off')  # 关闭坐标轴

    # # 自定义颜色映射：黑（0）→ 白（1）
    # black_white_cmap = LinearSegmentedColormap.from_list(
    #     'black_white',
    #     [(0, 0, 0), (1, 1, 1)],  # 0对应黑色，1对应白色
    #     N=256  # 颜色梯度数量
    # )

    # 归一化+增强对比度
    conf_np_normalized = (conf_np - conf_np.min()) / (conf_np.max() - conf_np.min() + 1e-8)

    # 绘制置信度图
    plt.subplot(122)
    conf_im = plt.imshow(conf_np_normalized, cmap='jet',vmin=0,vmax=1 )
    plt.title('置信度图（增强版）')
    plt.colorbar(conf_im, label='置信度')
    plt.axis('off')


    # 调整布局，避免重叠
    plt.tight_layout()
    
    # 保存图像
    save_path = "./output/vis_conf/visualization.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存前已完成绘图
    print(f"可视化结果已保存至: {save_path}")
    
    # 关闭图形，释放资源
    plt.close()
    ###############################
    return depth, conf