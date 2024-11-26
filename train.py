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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from scipy.spatial import cKDTree
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

def get_project_mask_fromviewcam(points, viewpoint_cam):
    R = viewpoint_cam.R
    T = viewpoint_cam.T
    FovY = viewpoint_cam.FoVy
    FovX = viewpoint_cam.FoVx
    H,W = viewpoint_cam.image_height, viewpoint_cam.image_width

    focal_y =  fov2focal(FovY, H)
    focal_x =  fov2focal(FovX, W)


    K = torch.tensor(
        [
            [focal_x, 0, W/2],
            [0, focal_y, H/2],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    mask = get_project_mask(points, R, T, K, H, W)
    return mask

def get_project_mask(points, R, T, K, H, W):
        '''
            Get the points mask projected on one view
            args:
                points: input points (N,3)  Tensor
                R: rotation matric (3,3) Tensor
                T: transference matric (3,1) Tensor
                K: intric matric (3,3) Tensor
                H,W: height & width of the image 
        '''
        w2v = getWorld2View2(R=R,t=T)
        R_w2v = torch.tensor(w2v[:3,:3], dtype=torch.float32)
        T_w2v = torch.tensor(w2v[:3,-1], dtype=torch.float32)

        point_c = (R_w2v[None,...] @ points[...,None])[...,0] + T_w2v

        point_c = point_c.to(torch.float32)

        point_c = point_c/torch.abs(point_c[...,-1][...,None])


        uv = (K[None,...] @ point_c[...,None])[...,0][...,:]

        mask = (uv[...,1]<  H) * (uv[...,0]<W) * (uv[...,1]>=0) * (uv[...,0]>=0) * uv[...,-1]>0
        return mask

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def mat_mul_2(a, b):  
    """
        perform matric multiply for symetric 3x3 metric and 3x1 vector
        args:
            a: matric parameters (N, 3, 3)
            b: vector parameters (N, 3)

    """

    c1 = a[...,0,:]*b
    c2 = a[...,1,:]*b
    c3 = a[...,2,:]*b

    return torch.concat((c1[...,None,:],c2[...,None,:],c3[...,None,:]), dim=-2)


def find_nearest_neibor(points):
    # points = torch.clone(points).to("cpu").numpy()
    tree = cKDTree(points)
    # print(points.shape[0])
    distances, indices = tree.query(points, k=2)

    nearest_neighbors = indices[:, 1]
    if np.max(nearest_neighbors) == points.shape[0]:
        idx = (np.where(nearest_neighbors==np.max(nearest_neighbors)))[0]
        mask = np.ones(points.shape[0])
        mask[idx] = 0
        mask = (mask==1)
        nearest_neighbors[idx] = idx

    else:
        mask = np.ones(points.shape[0])
        mask = (mask==1)

    return nearest_neighbors, mask

def get_directions(points, nearest_neighbors):
    direction = (points[nearest_neighbors] - points)/(np.linalg.norm(points[nearest_neighbors] - points,axis=-1)[...,None]+1e-8)
    return direction


def edge_aware_smooth_loss(rgb, depth, beita = 2,  grad_type="sobel"):
    if grad_type == "tv":
        rgb_grad_h = torch.pow(rgb[:,1:,:]-rgb[:,:-1,:], 2).mean()
        rgb_grad_w = torch.pow(rgb[:,:,1:]-rgb[:,:,:-1], 2).mean()
        depth_grad_h = torch.pow(depth[:,1:,:]-depth[:,:-1,:], 2).mean()
        depth_grad_w = torch.pow(depth[:,:,1:]-depth[:,:,:-1], 2).mean()
    elif grad_type == "sobel":
        temp_w = torch.tensor([[-1,0,1], [-2,0,2],[-1,0,1]],requires_grad=False, dtype=torch.float)[None,None,...].expand([1,3,3,3]).to(rgb.device)
        temp_h = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],requires_grad=False, dtype=torch.float)[None,None,...].expand([1,3,3,3]).to(rgb.device)
        rgb_grad_h = torch.abs(torch.nn.functional.conv2d(rgb[None,...],temp_h))
        rgb_grad_w = torch.abs(torch.nn.functional.conv2d(rgb[None,...],temp_w))

        depth_grad_h = torch.abs(torch.nn.functional.conv2d(depth[None,...],temp_h[:,:1,...]))
        depth_grad_w = torch.abs(torch.nn.functional.conv2d(depth[None,...],temp_w[:,:1,...]))

    loss_eas = torch.mean(depth_grad_h*torch.exp(-beita*rgb_grad_h)+depth_grad_w*torch.exp(-beita*rgb_grad_w))
    return loss_eas




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    mvs_n=0
    if args.train_mvs_prune:
        all_train_cams_stack  = scene.getTrainCameras().copy()
    else:
        all_train_cams_stack = None
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

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 3000 its we increase the levels of SH up to a maximum degree
        if iteration % 3000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


        if args.if_continue_reg:

                with torch.no_grad():

                    points = gaussians._xyz.to("cpu")
                    mask_points = get_project_mask_fromviewcam(points, viewpoint_cam)
                    points = points[mask_points]

                    neibor_idx , mask = find_nearest_neibor(points=points.numpy())

                    directions = torch.tensor(get_directions(points=points.numpy(), nearest_neighbors=neibor_idx))
                    # direction = directions.to(cov.device)
                    distance = torch.sqrt(torch.sum((points - points[neibor_idx])**2,dim=-1)).to("cuda")

                    Rotation = build_rotation(torch.nn.functional.normalize(gaussians._rotation))
                    Rotation = Rotation[mask_points]





                L = mat_mul_2(Rotation, torch.exp(gaussians._scaling)[mask_points])

                cov = torch.matmul(L , L.transpose(1, 2))
                nearest_neighbors = neibor_idx

                direction = directions.to(cov.device)

                projected_covariance_o = torch.matmul(torch.matmul(direction[:,None,...] , cov) , direction[...,None])
                projected_covariance_t = torch.matmul(torch.matmul(direction[:,None,...] , cov[nearest_neighbors,...]) , direction[...,None])


                sum_cov = (projected_covariance_o + projected_covariance_t)
                
                # #### used for LLFF dataset
                mask_thr = sum_cov[...,0,0]<distance/5
                mask = np.logical_and(mask_thr.to("cpu"), mask)

                if mask.sum()>0 :
                    loss+= args.continue_reg_weight * l1_loss(distance[mask], sum_cov[mask])
                # loss+= args.continue_reg_weight * l1_loss(distance[mask], sum_cov[mask])
        
        if args.if_bkg_occ:
            alfa = render_pkg["alpha"]
            gt = viewpoint_cam.original_image.cuda()
            # print(gt)
            # print(gt.shape)
            mask = (torch.mean(gt, dim=0)==1)
            # print(mask)
            # print(mask.shape)


            bg_loss = torch.mean(alfa[0,mask])
            loss+= args.occ_reg_weight*bg_loss

        # if args.occ_loss_weight>0:
        #     with torch.no_grad():
        #         points = gaussians._xyz.to("cpu")
        #         T = viewpoint_cam.T
        #         R = viewpoint_cam.R
        #         FovY = viewpoint_cam.FoVy
        #         FovX = viewpoint_cam.FoVx

        #         H,W = viewpoint_cam.original_image.shape[1:]

        #         focal_y =  fov2focal(FovY, H)
        #         focal_x =  fov2focal(FovX, W)

                
        #         K = torch.tensor(
        #             [
        #                 [focal_x, 0, W/2],
        #                 [0, focal_y, H/2],
        #                 [0, 0, 1],
        #             ],
        #             dtype=torch.float32,
        #         )

        #         w2v = getWorld2View2(R=R,t=T)
        #         R_w2v = torch.tensor(w2v[:3,:3], dtype=torch.float32)
        #         T_w2v = torch.tensor(w2v[:3,-1], dtype=torch.float32)

        #         point_c = (R_w2v[None,...] @ points[...,None])[...,0] + T_w2v

        #         point_c = point_c.to(torch.float32)

        #         distance_z = point_c[:,-1]
        #         # point_c = point_c/point_c[...,-1][...,None]
        #         point_c = point_c/torch.abs(point_c[...,-1][...,None])

        #         # point_c = point_c[point_c[...,-1]>0]

        #         # uv_l = (K[None,...] @ point_c[...,None])[...,0][...,:2]
        #         uv_l = (K[None,...] @ point_c[...,None])[...,0][...,:]
        #         # uv = torch.concat((uv,uv_l[None,...]),dim=0)

        #         # print((K[None,...] @ point_c[...,None])[...,0][...,-1])
        #         mask = (uv_l[:,1]<  H) * (uv_l[:,0]<W) * (uv_l[:,1]>=0) * (uv_l[:,0]>=0) * uv_l[:,-1]>0
        #         # print("occ mask shape:"+str(mask.shape[0]))
        #         # valid_distance = distance_z[mask]
        #         valid_distance = distance_z
        #         top_k_index = torch.argsort(valid_distance)[:args.occ_num]

        #         mask_prune = torch.zeros_like(valid_distance)
        #         mask_prune[top_k_index] = 1

        #         mask_prune = mask_prune.to(bool)

        #         gaussians.prune_points(mask_prune)
                # mask_dis = (valid_distance<6)
                # print(valid_distance[top_k_index])

            
            # loss_occ = torch.mean(torch.abs(gaussians._opacity[mask][top_k_index]))
            # loss+= args.occ_loss_weight*loss_occ

        if args.if_TV and iteration>args.tv_start:

            rendered_depth = render_pkg["depth"][0]

            loss_tv = edge_aware_smooth_loss(gt_image, rendered_depth.reshape((1,image.shape[1],image.shape[2])))

            loss+= args.tv_weight*loss_tv

            
            







        loss.backward()
        # print(gaussians._scaling.grad)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                    # if args.if_prune:
                    #     prune_mask = get_mvs_prune_mask(gaussians._xyz,cams=all_train_cams_stack,prune_thr=0.7)
                    # else:
                    #     prune_mask = None


                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    size_threshold = None
                    # if args.if_anneal:
                    #     freq_band = compute_freq_band(x, args.anneal_fs,  up_speed=args.anneal_speed)
                    #     x+= 1
                    # else:
                    #     freq_band = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold,  opt.prune_threshold, scene.cameras_extent, size_threshold)

                    if args.if_occ_prune:
                        gaussians.occ_prune(viewpoint_cam,args.occ_num)

                if iteration > opt.mvs_prune_start and iteration % opt.mvs_prune_iterval == 0 and iteration<opt.mvs_prune_end and args.train_mvs_prune:
                    mvs_n+=1
                    if mvs_n<=4:
                        # start_point = [0,64,128,256]  ## for blender
                        # mvs_prune_thresholds = [0.6, 0.65, 0.7, 0.8] ## for blender

                        start_point = [0,0,64,128]  ## for llff
                        mvs_prune_thresholds = [0.75, 0.8, 0.85, 0.85] ## for llff

                        # start_point = [0,0,64,128]  ## for llff
                        # mvs_prune_thresholds = [0.75, 0.8, 0.85, 0.85] ## for llff

                        # start_point = [0,0,64,128]  ## for 360
                        # mvs_prune_thresholds = [0.7, 0.75, 0.8, 0.8] ## for 360

                        # start_point = [0,0,64,64]  ## for dtu
                        # mvs_prune_thresholds = [0.4, 0.4, 0.5, 0.5] ## for dtu

                        mvs_prune_threshold = mvs_prune_thresholds[mvs_n-1]
                        feature_mask = torch.zeros(512)
                        feature_mask[start_point[4-mvs_n]:512] = 1
                    else:
                        feature_mask = torch.ones(512)
                        mvs_prune_threshold = args.mvs_prune_threshold
                    feature_mask = feature_mask.to(bool)
                    
                    # print(feature_mask)


                    gaussians.mvs_prune(all_train_cams_stack, mvs_prune_threshold, mask=feature_mask)
                    # gaussians.mvs_prune(all_train_cams_stack, args.mvs_prune_threshold, mask=feature_mask)
                    
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
            #             iteration > args.start_sample_pseudo:
            #         gaussians.reset_opacity()


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    np.random.seed(114514)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6079)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")