import numpy as np
import torch
import os
import torchvision
from torchvision.utils import save_image

def normalize(tensor):
    # 先分离张量，切断与计算图的连接
    tensor_detached = tensor.detach()
    tensor_cpu = tensor_detached.cpu()
    tensor_np = tensor_cpu.numpy()
    
    # 如果是布尔类型，转换为浮点数
    if tensor_np.dtype == bool:
        tensor_np = tensor_np.astype(np.float32)
    
    min_val, max_val = np.min(tensor_np), np.max(tensor_np)
    # 防止除零错误
    if max_val == min_val:
        return np.zeros_like(tensor_np)
    return (tensor_np - min_val) / (max_val - min_val)


def save_depth(render_depth,midas_depth,midas_conf,model_path,iteration,viewpoint_cam,beforegmm ):


    render_depth_path = os.path.join(model_path,"train" , "ours_{}".format(iteration), "{}_renders_depth".format(beforegmm))
    midas_depth_path = os.path.join(model_path, "train", "ours_{}".format(iteration), "{}_midas_depth".format(beforegmm))
    midas_conf_path=os.path.join(model_path, "train", "ours_{}".format(iteration), "{}_midas_conf".format(beforegmm))
    os.makedirs(render_depth_path, exist_ok=True)
    os.makedirs(midas_depth_path, exist_ok=True)  
    os.makedirs(midas_conf_path, exist_ok=True)
    render_depth_np=normalize(render_depth)
    midas_depth_np=normalize(midas_depth)
    midas_conf_np=normalize(midas_conf)

    render_depth_tensor = torch.from_numpy(render_depth_np).float()
    midas_depth_tensor = torch.from_numpy(midas_depth_np).float()
    midas_conf_tensor = torch.from_numpy(midas_conf_np).float()
    
    # 确保张量形状正确 (C, H, W)
    if render_depth_tensor.dim() == 2:
        render_depth_tensor = render_depth_tensor.unsqueeze(0)  # 添加通道维度
    if midas_depth_tensor.dim() == 2:
        midas_depth_tensor = midas_depth_tensor.unsqueeze(0)
    if midas_conf_tensor.dim() == 2:
        midas_conf_tensor = midas_conf_tensor.unsqueeze(0)

    torchvision.utils.save_image(render_depth_tensor, os.path.join(render_depth_path, viewpoint_cam.image_name + '.png'))
                                        #'{0:05d}'.format(idx) + ".png"))
    torchvision.utils.save_image(midas_depth_tensor, os.path.join(midas_depth_path, viewpoint_cam.image_name + ".png"))
    torchvision.utils.save_image(midas_conf_tensor, os.path.join(midas_conf_path, viewpoint_cam.image_name + ".png"))


# from utils.general_utils import vis_depth
        


