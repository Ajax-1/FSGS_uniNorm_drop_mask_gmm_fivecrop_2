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
import glob
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image
import imageio
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.general_utils import chamfer_dist
import numpy as np
import json
import cv2
import math
import torch
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
# from utils.depth_utils import estimate_depth
from torchvision import transforms
from mvs_modules.mvs_estimator import MvsEstimator

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    #bounds: np.array
    depth_image: np.array
    K: np.array = None
    bounds: np.array = None
    mvs_depth: np.array = None
    mvs_mask: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras2(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

        break

    def normalize(x):
        return x / np.linalg.norm(x)

    def viewmatrix(z, up, pos):
        vec2 = normalize(z)
        vec1_avg = up
        vec0 = normalize(np.cross(vec1_avg, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m


    c2w = np.concatenate([R, T], dim=1)
    print(c2w.shape)
    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    Num_views = 120
    rots = 2

    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]
    for theta in np.linspace(0., 2. * np.pi * rots, Num_views + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * 0.5), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))


    sys.stdout.write('\n')
    return cam_infos


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping,pcd=None, resolution=4, train_idx=None):
    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0., intr.params[1]],
                [0., focal_length_x, intr.params[2]],
                [0., 0., 1.]
            ], dtype=np.float32)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0., intr.params[2]],
                [0., focal_length_y, intr.params[3]],
                [0., 0., 1.]
            ], dtype=np.float32)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # read scene bounds
        pose_file_path = os.path.join(os.path.dirname(images_folder), 'poses_bounds.npy')
        poses_arr = np.load(pose_file_path)
        bds = poses_arr[extr.id-1, -2:] 
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        rgb_path = rgb_mapping[idx]   # os.path.join(images_folder, rgb_mapping[idx])
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        image = Image.open(rgb_path)
        #ajj711修改
        vggt_depth_path = os.path.join(images_folder, '..', 'vggt_depths', image_name + '_depth.npy')
        vggt_conf_path = os.path.join(images_folder, '..', 'vggt_depths', image_name + '_conf.npy')
        depth_image = np.load(vggt_depth_path) if os.path.exists(vggt_depth_path) else None
        conf_image = np.load(vggt_conf_path) if os.path.exists(vggt_conf_path) else None
        #ajj711修改结束
        '''
        path_depth_8=f"{path}/depth_8"
        depths_folder = os.path.join(path_depth_8, "depth_npy")
        depth_path = os.path.join(depths_folder, image_name + "_pred.npy")
        depth = None
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
        '''
        '''
        # depthmap, depth_weight = None, None
        # #depthmono = estimate_depth(image.cuda()).cpu().numpy()
        # depthloss = 1e8
        # if pcd is not None and idx in train_idx:
        #     depthmap, depth_weight = np.zeros((height//resolution,width//resolution)), np.zeros((height//resolution,width//resolution))
        #     K = np.array([[focal_length_x, 0, width//resolution/2],[0,focal_length_y,height//resolution/2],[0,0,1]])
        #     cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)) ### for coordinate definition, see getWorld2View2() function
        #     valid_idx = np.where(np.logical_and.reduce((cam_coord[2]>0, cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=width//resolution-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=height//resolution-1)))[0]
        #     pts_depths = cam_coord[-1:, valid_idx]
        #     cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx]
        #     depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = pts_depths
        #     depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = 1/pcd.errors[valid_idx] if pcd.errors is not None else 1
        #     depth_weight = depth_weight/depth_weight.max()

        #     # if model_zoe is None:
        #     #     model_zoe = torch.hub.load("./ZoeDepth", "ZoeD_NK", source="local", pretrained=True).to('cuda')
            
        #     # source_depth = model_zoe.infer_pil(image.convert("RGB"))
        #     #import pdb;pdb.set_trace()
            
        #     #source_depth =estimate_depth(image.cuda()).cpu().numpy()
        #     to_tensor = transforms.ToTensor()
        #     source_depth = estimate_depth(to_tensor(image).cuda()).cpu().numpy()
        #     target=depthmap.copy()
            
        #     target=((target != 0) * 255).astype(np.uint8)
        #     depthmap, depthloss = optimize_depth(source=source_depth, target=depthmap, mask=depthmap>0.0, depth_weight=depth_weight)
        '''   

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K,FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                image_name=image_name, width=width, height=height, mask=None, bounds=bds,depth_image=depth_image, conf_image=conf_image)
        #cam_info = CameraInfo(uid=uid, R=R, T=T, K=K,FovY=FovY, FovX=FovX, image=image, image_path=image_path,
        #        image_name=image_name, width=width, height=height, mask=None, bounds=bds, depth_image=depth)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.001):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    with torch.enable_grad():
        source = torch.from_numpy(source).cuda()
        target = torch.from_numpy(target).cuda()
        mask = torch.from_numpy(mask).cuda()
        depth_weight = torch.from_numpy(depth_weight).cuda()

        # Prune some depths considered "outlier"     
        with torch.no_grad():
            target_depth_sorted = target[target>1e-7].sort().values
            min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
            max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

            mask2 = target > min_prune_threshold
            mask3 = target < max_prune_threshold
            mask = torch.logical_and( torch.logical_and(mask, mask2), mask3)

        source_masked = source[mask]
        target_masked = target[mask]
        depth_weight_masked = depth_weight[mask]
        # tmin, tmax = target_masked.min(), target_masked.max()

        # # Normalize
        # target_masked = target_masked - tmin 
        # target_masked = target_masked / (tmax-tmin)

        scale = torch.ones(1).cuda().requires_grad_(True)
        shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

        optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
        loss = torch.ones(1).cuda() * 1e5

        iteration = 1
        loss_prev = 1e6
        loss_ema = 0.0
        
        while abs(loss_ema - loss_prev) > 1e-5:
            #optimizer.zero_grad()
            source_hat = scale*source_masked + shift
            loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

            # penalize depths not in [0,1]
            loss_hinge1 = loss_hinge2 = 0.0
            if (source_hat<=0.0).any():
                loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
            # if (source_hat>=1.0).any():
            #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
            
            loss = loss + loss_hinge1 + loss_hinge2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            iteration+=1
            if iteration % 1000 == 0:
                print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
                loss_prev = loss.item()
            loss_ema = loss.item() * 0.2 + loss_ema * 0.8

        loss = loss.item()
        print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss

def farthest_point_sampling(points, k):
    """
    Sample k points from input pointcloud data points using Farthest Point Sampling.

    Parameters:
    points: numpy.ndarray
        The input pointcloud data, a numpy array of shape (N, D) where N is the
        number of points and D is the dimensionality of each point.
    k: int
        The number of points to sample.

    Returns:
    sampled_points: numpy.ndarray
        The sampled pointcloud data, a numpy array of shape (k, D).
    """
    N, D = points.shape
    farthest_pts = np.zeros((k, D))
    distances = np.full(N, np.inf)
    farthest = np.random.randint(0, N)
    for i in range(k):
        farthest_pts[i] = points[farthest]
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return farthest_pts


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, n_views=0, dataset='LLFF',llffhold=8,mvs_config_path=None,  stage='train'):
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")

    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    pcd = fetchPly(ply_path)


    reading_dir = "images" if images == None else images
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    all_indices = list(range(len(cam_extrinsics)))
    if eval:
        train_idx = [idx for idx in all_indices if idx % llffhold != 0]
        if n_views >= 1:
            # 计算子采样索引
            train_cam_temp = [c for idx, c in enumerate(cam_extrinsics.values()) if idx % llffhold != 0]
            idx_sub = np.linspace(0, len(train_cam_temp) - 1, n_views)
            idx_sub = [round(i) for i in idx_sub]
            train_idx = [train_idx[i] for i in idx_sub]
    else:
        train_idx = all_indices
    
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir),  path=path, rgb_mapping=rgb_mapping,pcd=pcd,resolution=1,train_idx=train_idx)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        if dataset == 'NVSRGBD':
            print('Eval NVSRGBD Dataset!!!')
            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if 'train' in cam_info.image_name:
                    train_cam_infos.append(cam_info)
                    print('Train: ', cam_info.image_name)
                else:
                    test_cam_infos.append(cam_info)
                    print('Test: ', cam_info.image_name)
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if  stage == 'train':

        # get MVS depth, initial GS positions
        print('Predicting MVS depth...')
        mvs_estimator = MvsEstimator(mvs_config_path)
        vertices, mvs_depths, masks = mvs_estimator.get_mvs_pts(train_cam_infos)
        #import pdb;pdb.set_trace()
        torch.cuda.empty_cache()
        for i, cam in enumerate(train_cam_infos):
            # cam.mvs_depth = mvs_depths[i]
            # cam.mvs_mask = masks[i]
            new_caminfo = cam._replace(
            mvs_depth=mvs_depths[i],
            mvs_mask=masks[i])
            train_cam_infos[i] = new_caminfo
        #import pdb;pdb.set_trace()
        del mvs_estimator



    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        skip = 8 if transformsfile == 'transforms_test.json' else 1
        frames = contents["frames"][::skip]
        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            mask = norm_data[:, :, 3:4]
            if skip == 1:
                depth_image = np.load('../SparseNeRF/depth_midas_temp_DPT_Hybrid/Blender/' +
                                      image_path.split('/')[-4]+'/'+image_name+'_depth.npy')
            else:
                depth_image = None

            arr = cv2.resize(arr, (400, 400))
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            depth_image = None if depth_image is None else cv2.resize(depth_image, (400, 400))
            mask = None if mask is None else cv2.resize(mask, (400, 400))


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                        image_name=image_name, width=image.size[0], height=image.size[1],
                                        depth_image=depth_image, mask=mask))
    return cam_infos



def readNerfSyntheticInfo(path, white_background, eval, n_views=0, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    pseudo_cam_infos = train_cam_infos #train_cam_infos
    if n_views > 0:
        train_cam_infos = train_cam_infos[:n_views]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")

    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 30000
    #     print(f"Generating random point cloud ({num_pts})...")
    #
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    #
    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pseudo_cameras=pseudo_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
