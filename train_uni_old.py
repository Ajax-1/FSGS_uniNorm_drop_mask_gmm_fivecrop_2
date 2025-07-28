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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim,compute_multi_scale_loss_with_weights
from utils.depth_utils_uninorm_v4 import estimate_depth_uni
# from utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
from utils.save_depth import save_depth
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import torchvision
from torchvision.utils import save_image

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

def gmm_policy(scores, given_gt_thr=0.5, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        # import pdb; pdb.set_trace()
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr

def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, stage=pipe.stage,shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack,unseen_viewpoint_stack = None, None,None
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if not unseen_viewpoint_stack:
            unseen_viewpoint_stack = scene.getVirtualCameras().copy()
        
        
        if iteration % opt.reg_interval == 0 and iteration > args.start_sample_pseudo:
            viewpoint_cam = unseen_viewpoint_stack.pop(randint(0, len(unseen_viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,is_train=True,iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image_mask = torch.from_numpy(viewpoint_cam.mask).cuda()


        # Loss
        
        if not viewpoint_cam.is_virtual:
            Ll1 =  l1_loss_mask(image, gt_image)
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

            # import pdb;pdb.set_trace()
            rendered_depth = render_pkg["depth"][0]
            midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
            midas_conf=torch.tensor(viewpoint_cam.midas_conf).cuda()
            original_depth=rendered_depth
            original_shape=rendered_depth.shape
            if iteration %3100==0:
                # import pdb;pdb.set_trace()
                save_depth(rendered_depth,midas_depth,midas_conf,args.model_path,iteration,viewpoint_cam,beforegmm="beforegmm" )

            rendered_depth = rendered_depth.reshape(-1, 1)
            midas_depth = midas_depth.reshape(-1, 1)
            # import pdb;pdb.set_trace()
            
            midas_conf = midas_conf.reshape(-1, 1)
            if iteration %100 ==0:
                ### GMM
                conf_thre = gmm_policy(midas_conf)
                conf_mask = midas_conf>conf_thre
                rendered_depth = rendered_depth_filter=rendered_depth[conf_mask]
                midas_depth = midas_depth_filter=midas_depth[conf_mask]
                
                 
                if iteration %3100==0:

                    # import pdb;pdb.set_trace()
                    # 创建全零数组，用过滤后的值填充有效区域
                    rendered_depth_original = torch.zeros_like(original_depth).reshape(-1, 1)
                    midas_depth_original = torch.zeros_like(original_depth).reshape(-1, 1)
                    
                    # 将过滤后的值放回原位置
                    rendered_depth_original[conf_mask.reshape(-1, 1)] = rendered_depth_filter
                    midas_depth_original[conf_mask.reshape(-1, 1)] = midas_depth_filter

                    rendered_depth_original = rendered_depth_original.reshape(original_shape)
                    midas_depth_original = midas_depth_original.reshape(original_shape)
                    midas_conf_original = conf_mask.reshape(original_shape)
                    save_depth(rendered_depth_original,midas_depth_original,midas_conf_original,args.model_path,iteration,viewpoint_cam,beforegmm="aftergmm" )


            depth_loss = 1 - pearson_corrcoef( midas_depth, rendered_depth)
            '''
            depth_loss = min(
                            (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                            (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
            )
            '''
            loss += args.depth_weight * depth_loss

            if iteration > args.end_sample_pseudo:
                args.depth_weight = 0.001

            if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
                if not pseudo_stack:
                    pseudo_stack = scene.getPseudoCameras().copy()
                pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

                render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background,is_train=True,iteration=iteration)
                rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
                #midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

                midas_depth_pseudo, _ = estimate_depth_uni(render_pkg_pseudo["render"])

                #import pdb;pdb.set_trace()
                # from utils.depth_vis import depth_vis
                # import cv2
                # save_path='render_view_depth_0522'
                # depth_name_dpt='depth_dpt'
                # depth_name_mg='depth_mg'
                # depth_vis(save_path, depth_name_mg, midas_depth_pseudo_mg)
                # depth_vis(save_path, depth_name_dpt, midas_depth_pseudo)
                # cv2.imwrite('{}/render.jpg'.format(save_path), (render_pkg_pseudo["render"]*255).permute(1,2,0).cpu().detach().numpy())

                #conf_mask = midas_conf_pseudo>midas_conf_pseudo.mean()

                rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
                midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
                depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, midas_depth_pseudo)).mean()
                #depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

                if torch.isnan(depth_loss_pseudo).sum() == 0:
                    loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                    loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo
        else:

#             image[:,~gt_image_mask] = 0.0
# #============================以下是分块代码===========================================#            
#             unseen_v_ncc=compute_multi_scale_loss_with_weights(image, gt_image,crop_flag=args.crop_flag,ncc_weight=args.ncc_weight)
#             loss=args.unseen_v_ncc_weight*unseen_v_ncc
#             if loss==0:
#                 continue
#============================以下是L1代码===========================================#            
            # import pdb;pdb.set_trace()
            # image[:,~gt_image_mask] = 0.0
            unseen_v_Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.unseen_lambda_dssim) * unseen_v_Ll1 + opt.unseen_lambda_dssim * (1.0 - ssim(image, gt_image))

            

        loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
