#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from datetime import datetime
import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef
import os
import json
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
    
# transform1 = transforms.CenterCrop(( 352,480))
# # transform2 = transforms.CenterCrop(( 320,448))
# transform2 = transforms.CenterCrop(( 384,512))

##original
transform1 = transforms.FiveCrop((352, 480))
transform2 = transforms.FiveCrop((320, 448))

###for mipnerf360 1/8res
# transform1 = transforms.FiveCrop((256, 384))
# transform2 = transforms.FiveCrop((224, 352))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def depth_rank_loss(pred_depth, gt_depth, sample_n=512):
    pred_depth, gt_depth = pred_depth.view(-1), gt_depth.view(-1)
    selected_idxs = np.random.choice(pred_depth.shape[0], sample_n*2, replace=False)
    sample_idx0 = selected_idxs[:sample_n]
    sample_idx1 = selected_idxs[sample_n:]
    gt_depth0, gt_depth1 = gt_depth[sample_idx0], gt_depth[sample_idx1]
    pred_depth0, pred_depth1 = pred_depth[sample_idx0], pred_depth[sample_idx1]

    # note that we use inverse dpt mono depth
    mask = torch.where(gt_depth0 < gt_depth1, True, False)
    d0 = pred_depth0 - pred_depth1
    d1 = pred_depth1 - pred_depth0

    depth_loss0 = torch.zeros_like(pred_depth0)
    depth_loss0[mask] += d1[mask]
    depth_loss0[~mask] += d0[~mask]
    return torch.mean(torch.clamp(depth_loss0, min=0.0))

def pearson_corrcoef(x, y):
    import pdb;pdb.set_trace()
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
    if denominator == 0:
        return 0
    return numerator / denominator

def sliding_window_operation(rendered_depth, midas_depth, window_size=100, step_size=100, lambda_p=1.0):
    height, width = rendered_depth.shape[:2]
    total_loss = 0
    W = 0
    def adjust_lambda_p(loss, base_lambda_p):
        center = 0.5
        scale = 2.0 
        # tanh函数将损失映射到(-1,1)范围
        tanh_value = np.tanh(scale * (loss - center))     
        # 将tanh值映射到合适的权重范围
        adjusted_coef = base_lambda_p * (1 + 0.1 * tanh_value)
        return adjusted_coef
    # 计算常规窗口的最大索引
    max_y = ((height - window_size) // step_size) * step_size
    max_x = ((width - window_size) // step_size) * step_size
    # 处理规则网格的窗口
    for y in range(0, height - window_size + 1, step_size):
        for x in range(0, width - window_size + 1, step_size):
            window_rendered_depth = rendered_depth[y:y + window_size, x:x + window_size].reshape(-1, 1)
            window_midas_depth = midas_depth[y:y + window_size, x:x + window_size].reshape(-1, 1)
            loss_1 = 1 - pearson_corrcoef(255 - window_midas_depth, window_rendered_depth)
            loss_2 = 1 - pearson_corrcoef(1 / (window_midas_depth + 200.), window_rendered_depth)
            window_loss = min(loss_1, loss_2)

            adjusted_lambda_p = adjust_lambda_p(window_loss.item(), lambda_p)
            total_loss += adjusted_lambda_p * window_loss
            W += 1
    # 检查是否有未覆盖的右边缘区域
    if max_x + window_size < width:
        x = width - window_size  # 窗口右边界对齐图像右边界
        # 只处理常规网格未覆盖的右边缘
        for y in range(0, max_y + 1, step_size):
            window_rendered_depth = rendered_depth[y:y + window_size, x:x + window_size].reshape(-1, 1)
            window_midas_depth = midas_depth[y:y + window_size, x:x + window_size].reshape(-1, 1)
            loss_1 = 1 - pearson_corrcoef(255 - window_midas_depth, window_rendered_depth)
            loss_2 = 1 - pearson_corrcoef(1 / (window_midas_depth + 200.), window_rendered_depth)
            window_loss = min(loss_1, loss_2)
            adjusted_lambda_p = adjust_lambda_p(window_loss.item(), lambda_p)
            total_loss += adjusted_lambda_p * window_loss
            W += 1

    # 检查是否有未覆盖的底部边缘区域
    if max_y + window_size < height:
        y = height - window_size  # 窗口底边对齐图像底边
        # 只处理常规网格未覆盖的底部边缘
        for x in range(0, max_x + 1, step_size):
            window_rendered_depth = rendered_depth[y:y + window_size, x:x + window_size].reshape(-1, 1)
            window_midas_depth = midas_depth[y:y + window_size, x:x + window_size].reshape(-1, 1)

            loss_1 = 1 - pearson_corrcoef(255 - window_midas_depth, window_rendered_depth)
            loss_2 = 1 - pearson_corrcoef(1 / (window_midas_depth + 200.), window_rendered_depth)
            window_loss = min(loss_1, loss_2)

            adjusted_lambda_p = adjust_lambda_p(window_loss.item(), lambda_p)
            total_loss += adjusted_lambda_p * window_loss
            W += 1

    if W > 0:
        total_loss /= W
    return total_loss

def patchify(img, patch_size):
    img = img.unsqueeze(0)
    img = F.unfold(img, patch_size, stride=patch_size)
    img = img.transpose(2, 1).contiguous()
    return img.view(-1, patch_size, patch_size)

def patched_depth_ranking_loss(surf_depth, mono_depth, patch_size=-1, margin=1e-4):
    if patch_size > 0:
        surf_depth_patches = patchify(surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
        mono_depth_patches = patchify(mono_depth, patch_size).view(-1, patch_size * patch_size)
    else:
        surf_depth_patches = surf_depth.reshape(-1).unsqueeze(0)
        mono_depth_patches = mono_depth.reshape(-1).unsqueeze(0)

    length = (surf_depth_patches.shape[1]) // 2 * 2
    rand_indices = torch.randperm(length)
    surf_depth_patches_rand = surf_depth_patches[:, rand_indices]
    mono_depth_patches_rand = mono_depth_patches[:, rand_indices]
    
    patch_rank_loss = torch.max(
        torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * \
            (surf_depth_patches_rand[:, length // 2:] - surf_depth_patches_rand[:, :length // 2]) + margin,
        torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device)
    ).mean()


    #torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * (surf_depth_patches_rand[:, length // 2:] - surf_depth_patches_rand[:, :length // 2]) + margin, torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device)
    #torch.max(torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * (surf_depth_patches_rand[:, length // 2:] - surf_depth_patches_rand[:, :length // 2]) + margin, torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device))

    # patch_rank_loss1 = torch.max(
    # torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * \
    #     (surf_depth_patches_rand[:, :length // 2] - surf_depth_patches_rand[:, length // 2:]) + margin,
    # torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device)
    # ).mean()
    #import pdb;pdb.set_trace()
    return patch_rank_loss

def calculate_patch_means(image_tensor, patch_size):
    """
    按patch计算图像的均值
    :param image_tensor: 输入的图像张量，形状为 (C, H, W)
    :param patch_size: patch的大小，如 (height, width)
    :return: 每个patch的均值，形状为 (num_patches,)
    """
    C, H, W = image_tensor.shape
    patch_height, patch_width = patch_size

    # 计算patch的数量
    num_patches_h = H // patch_height
    num_patches_w = W // patch_width

    # 分割图像为patch
    patches = image_tensor.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(num_patches_h * num_patches_w, C, patch_height, patch_width)

    # 计算每个patch的均值
    patch_means = patches.mean(dim=(1, 2, 3))

    return patch_means

def inter_patched_depth_ranking_loss(surf_depth, mono_depth, patch_size=-1):
    surf_patch_mean = calculate_patch_means(surf_depth, patch_size=(patch_size, patch_size))
    mono_patch_mean = calculate_patch_means(mono_depth, patch_size=(patch_size, patch_size))

    # import pdb; pdb.set_trace()
    sample_n = surf_patch_mean.shape[0] // 2
    return depth_rank_loss(surf_patch_mean, mono_patch_mean, sample_n=sample_n)

def get_depth_ranking_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0

    for transform in [transform1, transform2]:
        # import pdb; pdb.set_trace()
        surf_depth_crops = transform(surf_depth)
        mono_depth_crops = transform(mono_depth.unsqueeze(0))

        ## yxp
        choice_idx = random.choice(range(len(surf_depth_crops)))
        surf_depth_crop = surf_depth_crops[choice_idx]
        mono_depth_crop = mono_depth_crops[choice_idx]

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = transform(object_mask)
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        depth_rank_loss += 0.5 * patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=32)
        #depth_rank_loss += 0.5 * patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=4)

        inter_patch_depth_rank_loss = inter_patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=32)

    return depth_rank_loss, inter_patch_depth_rank_loss

def normalize_depth(depth):
    min_val = torch.min(depth)
    max_val = torch.max(depth)
    if max_val - min_val > 0:
        normalized_depth = (depth - min_val) / (max_val - min_val)
    else:
        normalized_depth = torch.zeros_like(depth)
    return normalized_depth

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    #bs表示批量大小，tps表示每个图像块的总像素数
    
    ref=prepare_image_for_ncc(ref, patch_size=7,stride=4)
    nea=prepare_image_for_ncc(nea, patch_size=7,stride=4)

    bs, tps = nea.shape
    #图像块的边长
    patch_size = int(np.sqrt(tps))
    #数据预处理
    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def prepare_image_for_ncc(image, patch_size=7, stride=1):
    # 假设image的形状是(3, height, width)
    
    # 第一步：将RGB图像转换为灰度图像
    if image.shape[0] == 3:
        gray_image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    else:
        # 如果已经是单通道，直接使用
        gray_image = image.squeeze(0)
    
    # 添加批次和通道维度
    gray_image = gray_image.unsqueeze(0).unsqueeze(0)
    
    # 使用F.unfold高效提取补丁
    patches = F.unfold(gray_image, 
                      kernel_size=(patch_size, patch_size), 
                      stride=stride)
    
    # 重塑为[num_patches, patch_size*patch_size]
    patches_tensor = patches.squeeze(0).transpose(0, 1)
    return patches_tensor

def compute_fivecrop_loss(image, gt_image,crop_flag,  ncc_weight=1.0 ):
    
    if crop_flag==1:
        # import pdb;pdb.set_trace()
        for transform in [transform1, transform2]:
            # import pdb; pdb.set_trace()
            image_crops = transform(image)
            gt_crops = transform(gt_image)


        # ncc_weights = [1.0, 1.0, 1.0, 1.0, 1.5]
        ncc_weights = [ncc_weight] * 4 + [ncc_weight+0.05]
        losses = []

        for i, (img_crop, gt_crop) in enumerate(zip(image_crops, gt_crops)):
            # import pdb;pdb.set_trace()
            ncc, ncc_mask = lncc(img_crop, gt_crop)
            mask = ncc_mask.reshape(-1)
            ncc = ncc.reshape(-1)
            ncc = ncc[mask].squeeze()
            
            loss = 0
            if mask.sum() > 0:
                # 获取当前裁剪区域的权重（i为当前索引）
                weight = ncc_weights[i]
                ncc_loss = weight * ncc.mean()
                loss += ncc_loss
                losses.append(loss)
    else:
        # import pdb;pdb.set_trace()
        img_crop = image
        gt_crop = gt_image
        ncc, ncc_mask = lncc(img_crop, gt_crop)
        mask = ncc_mask.reshape(-1)
        ncc = ncc.reshape(-1)
        ncc = ncc[mask].squeeze()
        losses = []
        loss = 0
        if mask.sum() > 0:
            # 获取当前裁剪区域的权重（i为当前索引）
            ncc_weights = [ncc_weight] 
            weight = ncc_weights[0]
            ncc_loss = weight * ncc.mean()
            loss += ncc_loss
            losses.append(loss)
    # 返回总损失
    total_loss = sum(losses)
    return total_loss

def compute_multi_scale_loss_with_weights(image, gt_image, crop_flag, ncc_weight=1.0):
    loss = compute_fivecrop_loss(image, gt_image,crop_flag, ncc_weight)
    return loss    




# def multi_scale_center_crop_n_blocks(image, n_blocks,iteration):
    
#     channels, height, width = image.shape
#     crops = []
#     crop_sizes = []
#     save_dir = os.path.join('output', 'crop_images')
#     os.makedirs(save_dir, exist_ok=True)
#     # 生成n个尺度，从大到小均匀分布
#     scale_ratios = np.linspace(1.0, 0.1, n_blocks)
#     for i, scale in enumerate(scale_ratios):
#         crop_h = max(int(height * scale), 16)  # 确保最小尺寸不小于16
#         crop_w = max(int(width * scale), 16)
#         crop_h = min(int(height * scale), crop_h)
#         crop_w = max(int(width * scale), crop_w)
#         # 计算起始位置，确保从中心裁剪
#         start_h = (height - crop_h) // 2
#         start_w = (width - crop_w) // 2
#         # 执行裁剪
#         crop = image[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
#         crops.append(crop)
#         crop_sizes.append((crop_h, crop_w))
#     return crops, crop_sizes

# def adaptive_weight_assignment(losses, max_weight=0.5, min_weight=0.1):
#     losses = torch.tensor(losses) if not isinstance(losses, torch.Tensor) else losses
#     # 方法1：基于Z-score的自适应权重
#     mean_loss = losses.mean()
#     std_loss = losses.std() + 1e-8  # 避免除零
#     # 标准化损失
#     z_scores = (losses - mean_loss) / std_loss
#     # 使用sigmoid函数将z-score映射到权重
#     # 损失越小，z-score越负，sigmoid输出越接近1（权重越大）
#     weights = torch.sigmoid(-z_scores)
#     # 对权重进行平滑处理，避免极端值
#     weights = weights * 0.8 + 0.2 / len(weights)
#     # 限制权重范围
#     weights = torch.clamp(weights, min_weight, max_weight)
#     # 重新归一化以确保权重和为1
#     weights = weights / weights.sum()
#     return weights   



# def compute_multi_scale_center_loss(image, gt_image,iteration, n_blocks=4, ncc_weight=1.0, 
#                                    max_weight=0.5, min_weight=0.01):
#     # 从中心裁剪n个不同尺度的块
#     image_crops, crop_sizes = multi_scale_center_crop_n_blocks(image, n_blocks,iteration)
#     gt_crops, _ = multi_scale_center_crop_n_blocks(gt_image, n_blocks,iteration)
#     losses = []
#     for img_crop, gt_crop in zip(image_crops, gt_crops):
#         ncc, ncc_mask = lncc(img_crop, gt_crop)
#         mask = ncc_mask.reshape(-1)
#         #ncc = ncc.reshape(-1) * weights
#         ncc = ncc.reshape(-1)
#         ncc = ncc[mask].squeeze()
#         loss=0
#         if mask.sum() > 0:
#             ncc_loss = ncc_weight * ncc.mean()
#             loss += ncc_loss
#             losses.append(loss)
#     # 返回总损失
#     total_loss = sum(losses)
#     return total_loss
       


# def multi_scale_save_debug(image, n_blocks, iteration, path):
#     """
#     对图像进行多尺度裁剪并保存，增加详细调试信息解决全黑图像问题
    
#     Args:
#         image: 输入图像张量，形状为[C, H, W]
#         n_blocks: 需要裁剪的块数
#         iteration: 当前迭代次数，用于文件命名
#         path: 保存路径
#     """

#     channels, height, width = image.shape
#     crops = []
#     crop_sizes = []
#     save_dir = f'{path}'
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 打印图像基本信息，帮助调试
#     print(f"图像形状: {image.shape}")
    
#     # 检查整体图像是否有内容
#     if isinstance(image, torch.Tensor):
#         img_min = image.min().item()
#         img_max = image.max().item()
#     else:
#         img_min = np.min(image)
#         img_max = np.max(image)
    
#     print(f"整体图像值范围: min={img_min}, max={img_max}")
    
#     if img_max - img_min < 0.01:
#         print("警告: 整体图像几乎是全黑或单一颜色!")
    
#     # 直接保存整体图像，作为基准
#     try:
#         full_image = image
#         if isinstance(full_image, torch.Tensor):
#             if full_image.requires_grad:
#                 full_np = full_image.detach().cpu().numpy()
#             else:
#                 full_np = full_image.cpu().numpy()
#         else:
#             full_np = full_image
        
#         # 强制增强图像对比度
#         if full_np.dtype != np.uint8:
#             # 计算有效值范围 (使用百分位数以避免极端值的影响)
#             p_low = np.percentile(full_np, 1) if img_max > img_min else 0
#             p_high = np.percentile(full_np, 99) if img_max > img_min else 1.0
            
#             print(f"增强对比度范围: {p_low} 到 {p_high}")
            
#             # 线性拉伸对比度 (即使范围很小也拉伸)
#             if p_high > p_low:
#                 full_np = np.clip((full_np - p_low) / (p_high - p_low), 0, 1) * 255
#             else:
#                 # 如果实在没有值差异，则应用直方图均衡化
#                 full_np = (full_np - img_min) / (img_max - img_min + 1e-8) * 255
            
#             full_np = full_np.astype(np.uint8)
        
#         if channels == 1:
#             full_pil = Image.fromarray(full_np[0], mode='L')
#         else:
#             full_np_t = np.transpose(full_np, (1, 2, 0))
#             if channels == 3:
#                 full_pil = Image.fromarray(full_np_t.astype('uint8'), 'RGB')
#             elif channels == 4:
#                 full_pil = Image.fromarray(full_np_t.astype('uint8'), 'RGBA')
#             else:
#                 if channels == 1:
#                     full_np_t = np.repeat(full_np_t, 3, axis=2)
#                 elif channels > 3:
#                     full_np_t = full_np_t[:, :, :3]
#                 full_pil = Image.fromarray(full_np_t.astype('uint8'), 'RGB')
        
#         full_save_path = os.path.join(save_dir, f"iteration_{iteration}_full_image.png")
#         full_pil.save(full_save_path)
#         print(f"成功保存完整图像: {full_save_path}")
#     except Exception as e:
#         print(f"保存完整图像时出错: {str(e)}")
#         import traceback
#         traceback.print_exc()
    
#     # 生成n个尺度，从大到小均匀分布
#     scale_ratios = np.linspace(1.0, 0.1, n_blocks)
    
#     # 查找图像中非零区域
#     if isinstance(image, torch.Tensor):
#         # 查找每个通道中的非零区域
#         nonzero_mask = torch.sum(image > 0.01, dim=0) > 0  # 保守的阈值
#         if nonzero_mask.sum() > 0:
#             nonzero_indices = torch.nonzero(nonzero_mask)
#             if len(nonzero_indices) > 0:
#                 min_h, min_w = nonzero_indices.min(dim=0).values.cpu().numpy()
#                 max_h, max_w = nonzero_indices.max(dim=0).values.cpu().numpy()
#                 print(f"找到非零区域: 从 ({min_h},{min_w}) 到 ({max_h},{max_w})")
#             else:
#                 min_h, min_w, max_h, max_w = 0, 0, height, width
#         else:
#             # 没有找到非零区域，使用整个图像
#             min_h, min_w, max_h, max_w = 0, 0, height, width
#     else:
#         # NumPy版本
#         nonzero_mask = np.sum(image > 0.01, axis=0) > 0
#         if np.any(nonzero_mask):
#             nonzero_indices = np.argwhere(nonzero_mask)
#             min_h, min_w = nonzero_indices.min(axis=0)
#             max_h, max_w = nonzero_indices.max(axis=0)
#             print(f"找到非零区域: 从 ({min_h},{min_w}) 到 ({max_h},{max_w})")
#         else:
#             min_h, min_w, max_h, max_w = 0, 0, height, width
    
#     # 确保找到的区域有一定大小
#     if max_h - min_h < 10 or max_w - min_w < 10:
#         print("警告: 找到的非零区域太小，使用整个图像")
#         min_h, min_w, max_h, max_w = 0, 0, height, width
    
#     # 遍历每个尺度
#     for i, scale in enumerate(scale_ratios):
#         # 计算裁剪尺寸，确保最小尺寸不小于16
#         crop_h = max(int(height * scale), 16)
#         crop_w = max(int(width * scale), 16)
        
#         # 确保尺寸不超过原图
#         crop_h = min(crop_h, height)
#         crop_w = min(crop_w, width)
        
#         # 计算起始位置，尝试包含找到的非零区域
#         if max_h - min_h > 0 and max_w - min_w > 0:
#             # 中心点
#             center_h = (min_h + max_h) // 2
#             center_w = (min_w + max_w) // 2
            
#             # 计算裁剪起始点，使非零区域尽可能居中
#             start_h = max(0, center_h - crop_h // 2)
#             start_w = max(0, center_w - crop_w // 2)
            
#             # 确保不会超出边界
#             if start_h + crop_h > height:
#                 start_h = height - crop_h
#             if start_w + crop_w > width:
#                 start_w = width - crop_w
#         else:
#             # 如果没有明确的非零区域，则使用中心裁剪
#             start_h = (height - crop_h) // 2
#             start_w = (width - crop_w) // 2
        
#         print(f"裁剪 {i}: 尺寸={crop_w}x{crop_h}, 起始位置=({start_h},{start_w})")
        
#         # 执行裁剪
#         crop = image[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
#         crops.append(crop)
#         crop_sizes.append((crop_h, crop_w))
        
#         # 将张量转换为numpy数组用于保存
#         if isinstance(crop, torch.Tensor):
#             if crop.requires_grad:
#                 crop_np = crop.detach().cpu().numpy()
#             else:
#                 crop_np = crop.cpu().numpy()
#         else:
#             crop_np = crop
        
#         # 检查裁剪区域的值范围
#         crop_min = np.min(crop_np)
#         crop_max = np.max(crop_np)
#         print(f"裁剪 {i} 值范围: min={crop_min}, max={crop_max}")
        
#         # 检查是否有NaN值并替换
#         if np.isnan(crop_np).any():
#             print(f"警告: 裁剪 {i} 中检测到NaN值，替换为零")
#             crop_np = np.nan_to_num(crop_np)
        
#         # 即使对比度很低也保存，但应用强制对比度增强
#         if crop_np.dtype != np.uint8:
#             # 使用百分位数计算对比度拉伸范围
#             p_low = np.percentile(crop_np, 1) if crop_max > crop_min else 0
#             p_high = np.percentile(crop_np, 99) if crop_max > crop_min else 1.0
            
#             # 强制拉伸对比度 (即使范围很小)
#             if p_high > p_low:
#                 crop_np = np.clip((crop_np - p_low) / (p_high - p_low), 0, 1) * 255
#             else:
#                 # 如果实在没有值差异，使图像灰色而不是黑色
#                 crop_np = np.ones_like(crop_np) * 128
            
#             crop_np = crop_np.astype(np.uint8)
        
#         # 转换裁剪图像为PIL图像并保存
#         try:
#             if channels == 1:
#                 # 单通道图像
#                 pil_image = Image.fromarray(crop_np[0], mode='L')
#             else:
#                 # 多通道图像
#                 crop_np = np.transpose(crop_np, (1, 2, 0))
                
#                 # 确保通道数正确
#                 if channels == 3:
#                     pil_image = Image.fromarray(crop_np.astype('uint8'), 'RGB')
#                 elif channels == 4:
#                     pil_image = Image.fromarray(crop_np.astype('uint8'), 'RGBA')
#                 else:
#                     # 处理其他通道数的情况
#                     if channels == 1:
#                         crop_np = np.repeat(crop_np[:, :, np.newaxis], 3, axis=2)
#                     elif channels > 3:
#                         crop_np = crop_np[:, :, :3]
#                     pil_image = Image.fromarray(crop_np.astype('uint8'), 'RGB')
            
#             # 定义保存路径，带有迭代信息
#             save_path = os.path.join(save_dir, f"iteration_{iteration}_crop_{i}_size_{crop_w}x{crop_h}.png")
#             pil_image.save(save_path)
#             print(f"成功保存图像: {save_path}")
#         except Exception as e:
#             print(f"警告: 无法保存图像 {i}，迭代 {iteration}. 错误: {str(e)}")
#             import traceback
#             traceback.print_exc()
    
#     return crops, crop_sizes



