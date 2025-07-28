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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff, generate_pseudo_poses_llff,generate_random_poses_360
from scene.cameras import PseudoCamera,Camera
from utils.warp_utils import Warper
import torch
from tqdm import tqdm
from utils.graphics_utils import getWorld2View, focal2fov
import cv2
import torchvision

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0],stage='render'):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.args = args
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.stage = stage

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        self.virtual_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views,mvs_config_path=args.mvs_config, 
                                                          stage=self.stage,dataset=args.dataset)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            pseudo_cams = []
            if args.source_path.find('llff'):
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            elif args.source_path.find('360'):
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            view = self.train_cameras[resolution_scale][0]
            for pose in pseudo_poses:
                pseudo_cams.append(PseudoCamera(
                    R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height
                ))
            self.pseudo_cameras[resolution_scale] = pseudo_cams
        
        if self.stage == 'train':
            print("Generating Virtual Cameras, num: ", args.total_virtual_num)
            # if args.dataset == 'DTU':
            #     self.virtual_cameras[resolution_scale] = self.generateVirtualCams(self.train_cameras[resolution_scale], v_num=args.total_virtual_num, use_mask=True)
            # else:
            # #生成其他数据集，如LLFF的虚拟相机
            self.virtual_cameras[resolution_scale] = self.generateVirtualCams(self.train_cameras[resolution_scale], v_num=args.total_virtual_num, inpaint=True)
            torch.cuda.empty_cache()


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]
    
    def getVirtualCameras(self, scale=1.0):
        return self.virtual_cameras[scale]
 #        #创建虚拟相机对象
    def generateVirtualCams(self, input_cams, v_num=120, batch_size=24, use_mask=False, inpaint=False):
        print('训练视图数量: ', len(input_cams))
        #准备生成虚拟视角所需的数据
        #import pdb;pdb.set_trace()
        input_imgs, mvs_depths, masks, input_extrs, input_intrs, \
        target_extrs, target_intrs = self.prepare_data(input_cams, v_num)
        #import pdb;pdb.set_trace()
        # 分批处理
        input_batches = create_batches(input_imgs, mvs_depths, masks, input_extrs, input_intrs, batch_size=batch_size)
        target_batches = create_batches(target_intrs, target_extrs, batch_size=batch_size)
        #初始化warper类
        warper = Warper()
        #存储变形后的帧、有效掩码、变形后的深度
        warped_frames, valid_masks, warped_depths = [], [], []
        with torch.no_grad():
            for (input_imgs_batch, mvs_depths_batch, masks_batch, input_extrs_batch, input_intrs_batch), (target_intrs_batch, target_extrs_batch) \
                in tqdm(zip(input_batches, target_batches), desc="生成未见视图先验", unit="batch", total=int(v_num / batch_size)):
                torch.cuda.empty_cache()
                masks_batch = None if not use_mask else masks_batch
                # 通过前向变形获取未见视图的先验
                warped_frame, valid_mask, warped_depth, _ = warper.forward_warp(input_imgs_batch, masks_batch, mvs_depths_batch, input_extrs_batch, 
                                                                                target_extrs_batch, input_intrs_batch, target_intrs_batch)
                warped_frames.append(warped_frame.cpu())
                valid_masks.append(valid_mask.cpu())
                warped_depths.append(warped_depth.cpu())

        warped_depths = torch.cat(warped_depths, dim=0)
        valid_masks = torch.cat(valid_masks, dim=0)
        warped_frames = torch.cat(warped_frames, dim=0)
        
        virtual_cams = []
        
        # 检查是否应该使用图像修复并在需要时初始化
        use_inpainting = inpaint
        simple_lama = None
        if use_inpainting:
            try:
                # 尝试导入SimpleLama模块
                from simple_lama_inpainting import SimpleLama
                simple_lama = SimpleLama()
                # # 在小样本上测试图像修复是否有效
                # test_img = np.zeros((64, 64, 3), dtype=np.float32)
                # test_mask = np.ones((64, 64), dtype=np.uint8)
                # try:
                #     simple_lama(test_img, test_mask)
                #     print("图像修复测试成功，将使用图像修复")
                # except Exception as e:
                #     print(f"图像修复测试失败，禁用图像修复: {e}")
                #     use_inpainting = False
            except ImportError:
                print("无法导入SimpleLama，禁用图像修复")
                use_inpainting = False
        
        # 使用洞周围有效像素的简单平均值填充洞的函数
        def simple_hole_filling(img, mask, kernel_size=5):
            """使用有效像素的平均值进行简单的洞填充"""
            img_np = img.permute(1, 2, 0).cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy().astype(bool)
            
            # 创建填充后的图像
            filled_img = img_np.copy()
            
            # 稍微扩张有效区域以帮助填充
            
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_mask = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1).astype(bool)
            
            # 对每个通道进行处理
            for c in range(img_np.shape[2]):
                # 获取有效像素值
                valid_values = img_np[:, :, c][dilated_mask]
                if len(valid_values) > 0:
                    # 用有效像素的平均值填充洞
                    filled_img[:, :, c][~mask_np] = valid_values.mean()
            
            return torch.from_numpy(filled_img).permute(2, 0, 1)
        
        # 遍历处理每个虚拟相机
        for i in range(v_num):
            # 生成虚拟相机的ID
            id = len(input_cams) + i
            # 获取目标相机的旋转矩阵和平移向量
            R, T = target_extrs[i, :3, :3].cpu().numpy().transpose(), target_extrs[i, :3, 3].cpu().numpy()
            # 计算焦距
            focal_length_x, focal_length_y = target_intrs[i][0,0], target_intrs[i][1,1]
            # 计算图像尺寸
            H, W = warped_frames.shape[2:4]
            # 计算垂直和水平视场角
            FovY = focal2fov(focal_length_y, H)
            FovX = focal2fov(focal_length_x, W)
            # 获取变形后的图像
            warped_img = warped_frames[i]
            mask = valid_masks[i].squeeze().to(torch.bool).detach().cpu().numpy()
            
            save_vir_path = 'vir_0508'
            save_vir_inpaint_path = 'vir_inpaint_0508'
            if not os.path.exists(save_vir_path):
                os.makedirs(save_vir_path)
                os.makedirs(save_vir_inpaint_path)
            
            torchvision.utils.save_image(warped_img, os.path.join(save_vir_path, str(i) + '.png'))
            torchvision.utils.save_image(valid_masks[i].squeeze(), os.path.join(save_vir_path, str(i) + '_mask.png'))

            # import pdb; pdb.set_trace()
            # 根据可用方法处理图像
            if use_inpainting and simple_lama is not None:
                warped_img = simple_hole_filling(warped_img, torch.from_numpy(mask))
            
                torchvision.utils.save_image(warped_img, os.path.join(save_vir_inpaint_path, str(i) + '.png'))
            
            # Optional: Load VGGT depth for this virtual view if precomputed ajj711修改
            vggt_depth_path = os.path.join('vggt_virtual_depths', f'virtual_{i}_depth.npy')
            if os.path.exists(vggt_depth_path):
                warped_depths[i] = torch.from_numpy(np.load(vggt_depth_path))
            #ajj711修改结束
            virtual_cam = Camera(colmap_id=None, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=warped_img, gt_alpha_mask=None,
                                image_name='virtual_'+str(i), uid=id, data_device='cpu',
                                K=target_intrs[i].cpu().numpy(), mvs_depth=warped_depths[i].cpu().numpy(), mask=mask, is_virtual=True)#mask=valid_masks[i].cpu().numpy()
            virtual_cams.append(virtual_cam)

        
        return virtual_cams


    def prepare_data(self, input_cams, v_num):
        # choose one source view randomly
        ids = np.random.choice(len(input_cams), v_num, replace=True)
            
        input_imgs = torch.stack([input_cams[id].original_image for id in ids])
        mvs_depths = torch.from_numpy(np.stack([input_cams[id].mvs_depth for id in ids])).unsqueeze(1)
        masks = torch.from_numpy(np.stack([input_cams[id].mask for id in ids])).unsqueeze(1)
        input_extrs = torch.from_numpy(np.stack([getWorld2View(input_cams[id].R, input_cams[id].T) for id in ids]))
        input_intrs = torch.from_numpy(np.stack([input_cams[id].K for id in ids]))

        # generate random poses for unseen views
        if self.args.dataset == 'LLFF' or self.args.dataset == 'Tank' or self.args.dataset == 'NVSRGBD':
            bds = np.stack([cam.bds for cam in input_cams])
            target_poses = torch.from_numpy(generate_pseudo_poses_llff(input_extrs, bds, n_poses=v_num))
        # elif self.args.dataset == 'DTU':
        #     target_poses = torch.from_numpy(generate_random_poses_dtu(input_extrs, n_poses=v_num))
        ######mvpgs没有跑mip360数据集，所以下面这个位姿生成是按照FSGS改的
        elif self.args.dataset =="mipnerf360":
            selected_cams = [input_cams[id] for id in ids]
            poses_list = generate_random_poses_360(selected_cams, n_frames=v_num+1)
            # import pdb;pdb.set_trace()
            target_poses = torch.from_numpy(np.array(poses_list))
        else:
            raise ValueError("Unknown dataset: {}".format(self.args.dataset))

        target_extrs = torch.from_numpy(np.stack([np.linalg.inv(target_poses[i]) for i in range(target_poses.shape[0])]))
        target_intrs = torch.from_numpy(np.stack([input_cams[0].K] * len(ids)))  # same intrinsics

        return input_imgs.cpu(), mvs_depths.cpu(), masks.cpu(), input_extrs, input_intrs, target_extrs, target_intrs

def create_batches(*tensors: torch.Tensor, batch_size: int):
    return list(zip(*[torch.split(tensor, batch_size) for tensor in tensors]))