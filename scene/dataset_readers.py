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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch.nn.functional as F
from tqdm import tqdm
from utils.camera_utils import get_img_feature
def mvs_prune(train_cam_infos , points,  res , prune_thr=0.8, mask=None, chunk_size = 2000):

    def feature_sim(feature0,feature1):
        f0 = feature0
        f1 = feature1
        sim_map = torch.sum(f0*f1,dim=0)/(torch.norm(f0,dim=0)*torch.norm(f1,dim=0))
        return sim_map


    prune_masks = np.array([])

    images_origin = [np.array(infos.image) for infos in train_cam_infos]
    orig_h, orig_w = images_origin[0].shape[:2]


    resolution = (int(orig_w / res), int(orig_h / res))
    images = [PILtoTorch(infos.image, resolution) for infos in train_cam_infos]     

    T = [torch.tensor(infos.T,dtype = torch.float) for infos in train_cam_infos]
    R = [torch.tensor(infos.R,dtype = torch.float) for infos in train_cam_infos]
    FovY = [infos.FovY for infos in train_cam_infos]
    FovX = [infos.FovX for infos in train_cam_infos]

    H,W = images[0].shape[1:]

    focal_y =  fov2focal(FovY[0], H)
    focal_x =  fov2focal(FovX[0], W)

    K = torch.tensor(
        [
            [focal_x, 0, W/2],
            [0, focal_y, H/2],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    feature_extract_model = resnet50
    feature_map = torch.tensor([])

    for idx, img in (enumerate(images)):

        img_feature = get_img_feature(img[None,...],model=feature_extract_model)

        feature_map = torch.concat((feature_map,img_feature[None,...]),dim=0)

    for i in range(0, points.shape[0], chunk_size):
        points_chunk = points[i:i+chunk_size]

        uv = torch.tensor([])
        for idx, img in enumerate(images):

            w2v = getWorld2View2(R=R[idx].numpy(),t=T[idx].numpy())
            R_w2v = torch.tensor(w2v[:3,:3], dtype=torch.float32)
            T_w2v = torch.tensor(w2v[:3,-1], dtype=torch.float32)

            point_c = (R_w2v[None,...] @ points_chunk[...,None])[...,0] + T_w2v

            point_c = point_c.to(torch.float32)
            point_c = point_c/np.abs(point_c[...,-1][...,None])

            uv_l = (K[None,...] @ point_c[...,None])[...,0][...,:]
            uv = torch.concat((uv,uv_l[None,...]),dim=0)

        mask_p = (uv[:,:,1]<  H) * (uv[:,:,0]<W) * (uv[:,:,1]>=0) * (uv[:,:,0]>=0) * uv[:,:,-1]>0
        uv = uv[...,:2]

        tau = prune_thr
        uv = uv.to(torch.int64)
        prune_mask = []
        max_sims = []
        for i in range(mask_p.shape[1]):
            ### feature convolution
            max_sim = 0
            ###
            selected_feature = feature_map[mask_p[:,i],:,uv[mask_p[:,i],i,1],uv[mask_p[:,i],i,0]]
            prune_flag = True
            if selected_feature is not None:
                if selected_feature.shape[0]>1:
                    for f_i in range(selected_feature.shape[0]):
                        for j in range(f_i+1,selected_feature.shape[0]):
                            sim_f = feature_sim(selected_feature[f_i],selected_feature[j])

                            if sim_f > tau:
                                prune_flag = False
                            
                            if sim_f > max_sim:
                                max_sim = sim_f
            max_sims.append(max_sim)
            prune_mask.append(prune_flag)


        if len(max_sims)>0:
            prune_mask = np.stack(max_sims)<np.stack(max_sims).mean()
        else:
            prune_mask = np.array([])
        prune_masks = np.concatenate((prune_masks, prune_mask), axis=0)


    return torch.tensor(prune_masks).to(bool)





from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import match_pair
import copy 
import torch.nn.functional as F
import torch
from utils.general_utils import PILtoTorch

def get_sparse_correspondences(train_cam_infos, res, tau = 1000):

    def compute_midpoint_between_lines_batch(P1, u1, P2, u2):
        sub = torch.sum(u2*u2,axis=-1)*torch.sum(u1*u1,axis=-1)-torch.sum(u1*u2,axis=-1)*torch.sum(u1*u2,axis=-1)
        n = (torch.sum(u1*u2,axis=-1)*torch.sum(u1*P2,axis=-1)-torch.sum(u1*u2,axis=-1)*torch.sum(u1*P1,axis=-1)-torch.sum(u1*u1,axis=-1)*torch.sum(u2*P2,axis=-1)+torch.sum(u1*u1,axis=-1)*torch.sum(u2*P1,axis=-1))/(sub+1e-8)
        m = (torch.sum(u2*u2,axis=-1)*torch.sum(u1*P2,axis=-1)-torch.sum(u2*u2,axis=-1)*torch.sum(u1*P1,axis=-1)+torch.sum(u1*u2,axis=-1)*torch.sum(u2*P1,axis=-1)-torch.sum(u1*u2,axis=-1)*torch.sum(u2*P2,axis=-1))/(sub+1e-8)
        Dis = np.linalg.norm((P1+m[:,None]*u1-P2-n[:,None]*u2),axis=-1)
        mid_point = (P1+m[:,None]*u1+P2+n[:,None]*u2)/2
        
        return mid_point, Dis



    def get_pairs(R, T, pairs):
        for i in range(len(T)):
            dist = torch.inf 
            for j in range(len(T)):
                if j!=i: #  and pairs[j] !=i
                    new_dist = torch.sqrt(torch.sum((T[i] - T[j])**2))
                     
                    RT0 = getWorld2View2(R=R[i].numpy(),t=T[i].numpy())
                    RT1 = getWorld2View2(R=R[j].numpy(),t=T[j].numpy())

                    C2W0 = np.linalg.inv(RT0)
                    C2W1 = np.linalg.inv(RT1)
                    dir_z = torch.tensor([0. ,0. ,1.])
                    R0 = torch.tensor(C2W0[:3,:3])
                    R1 = torch.tensor(C2W1[:3,:3])
                    dir_0 = (R0 @ dir_z[...,None])[...,0]
                    dir_1 = (R1 @ dir_z[...,None])[...,0]

                    if new_dist <  dist and torch.sum(dir_0*dir_1)>0:
                        # if pairs[j] !=i:
                            dist = new_dist
                            pairs[i] = j
            if pairs[pairs[i]] == i:
                pairs[i] = -1
        return pairs
    
    def get_pointcloud(pix0,pix1,R,T,K,W,H):
        x, y = torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            indexing="xy",
        )
        directions = F.pad(
            torch.stack(
                [
                    (x - K[0, 2] + 0.5) / K[0, 0],
                    (y - K[1, 2] + 0.5) / K[1, 1] * 1,
                ],
                dim=-1,
            ),
            (0, 1),
            value=1,
        ) 

        RT0 = getWorld2View2(R=R[0].numpy(),t=T[0].numpy())
        RT1 = getWorld2View2(R=R[1].numpy(),t=T[1].numpy())

        C2W0 = np.linalg.inv(RT0)
        C2W1 = np.linalg.inv(RT1)

        R0 = torch.tensor(C2W0[:3,:3])
        R1 = torch.tensor(C2W1[:3,:3])
        T0 = torch.tensor(C2W0[:3,-1])
        T1 = torch.tensor(C2W1[:3,-1])


        directions_pix0 =  ((R0) @ directions[pix0[...,0],pix0[...,1],:][...,None])[...,0]
        directions_pix1 =  ((R1) @ directions[pix1[...,0],pix1[...,1],:][...,None])[...,0]

        directions_pix0 = directions_pix0 / np.linalg.norm(directions_pix0, axis=-1, keepdims=True)
        directions_pix1 = directions_pix1 / np.linalg.norm(directions_pix1, axis=-1, keepdims=True)

        origins_pix0 = T0
        origins_pix1 = T1

        points, dis = compute_midpoint_between_lines_batch(origins_pix0,directions_pix0,origins_pix1,directions_pix1)
        mask = dis<tau
        points = points[mask]
        print("filtered num:"+str(mask.sum()))

        return points, mask


    images_origin = [np.array(infos.image) for infos in train_cam_infos]
    orig_h, orig_w = images_origin[0].shape[:2]

    resolution = (int(orig_w / res), int(orig_h / res))

    images = [PILtoTorch(infos.image, resolution) for infos in train_cam_infos]     

    T = [torch.tensor(infos.T,dtype = torch.float) for infos in train_cam_infos]
    R = [torch.tensor(infos.R,dtype = torch.float) for infos in train_cam_infos]
    FovY = [infos.FovY for infos in train_cam_infos]
    FovX = [infos.FovX for infos in train_cam_infos]
    pairs = np.arange(len(T))


    pairs = get_pairs(R, T,pairs)

    N = len(images)

    H,W = images[0].shape[1:]

    focal_y =  fov2focal(FovY[0], H)
    focal_x =  fov2focal(FovX[0], W)

    K = torch.tensor(
        [
            [focal_x, 0, W/2],
            [0, focal_y, H/2],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    ) 

    extractor = SuperPoint(max_num_keypoints=2048).eval()#.cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval()#.cuda()  # load the matcher
    point_cloud = torch.tensor([])
    color = torch.tensor([])
    for i in range(pairs.shape[0]):
        current_idx = i
        corres_idx = pairs[i]
        if current_idx == corres_idx or corres_idx == -1:
            continue

        feats0, feats1, matches01 = match_pair(extractor, matcher, images[current_idx], images[corres_idx])
        
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        
        pix0 = np.round(points0).to(int)
        mid = copy.deepcopy(pix0[...,0])
        pix0[...,0] = copy.deepcopy(pix0[...,1])
        pix0[...,1] = mid
        pix1 = np.round(points1).to(int)
        mid = copy.deepcopy(pix1[...,0])
        pix1[...,0] = copy.deepcopy(pix1[...,1])
        pix1[...,1] = mid


        pix0 = torch.tensor(pix0)
        pix1 = torch.tensor(pix1)


        R_in = torch.stack([R[i],R[pairs[i]]])
        T_in = torch.stack([T[i],T[pairs[i]]])


        point_cloud_new, mask = get_pointcloud(pix0, pix1, R = R_in, T = T_in, K = K, W = W, H = H)
        point_cloud = torch.concat((point_cloud,point_cloud_new),dim=0)

        color_new = (images[current_idx][:,pix0[...,0],pix0[...,1]].permute(1,0) + images[corres_idx][:,pix1[...,0],pix1[...,1]].permute(1,0))/2

        color_new = color_new[mask]
        color = torch.concat((color,color_new),dim=0)


    return point_cloud, color



def generate_spiral_path(poses, bounds, fix_rot, n_frames=120, n_rots=1, zrate=.5):
    def poses_avg(poses):
        """New pose using average position, z-axis, and up vector of input poses."""
        position = poses[:, :3, 3].mean(0)
        z_axis = poses[:, :3, 2].mean(0)
        up = poses[:, :3, 1].mean(0)
        cam2world = viewmatrix(z_axis, up, position)
        return cam2world

    def viewmatrix(lookdir, up, position, subtract_position=False):
        """Construct lookat view matrix."""
        vec2 = normalize((lookdir - position) if subtract_position else lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.]])

    print(focal, radii)

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, focal, 1.]
        z_axis = -position + lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses



class octree_node():
    def __init__(self, bound, res):
        self.bound = bound
        self.size = np.mean(bound[1] - bound[0])
        self.resolution = res
        self.occ_node = []
        self.unocc_node = []
        self.points_idx = []

        self.neighbor = []

    def add_occ_leaves(self, node):
        self.occ_node.append(node)


    def add_unocc_leaves(self, node):
        self.unocc_node.append(node)

    def add_points(self, points_idx):
        self.points_idx+= points_idx



class octree:
    def __init__(self, point_cloud, bound, resolution):
        self.point_cloud = point_cloud
        self.bound = bound
        self.resolution = resolution


        mask_points = (np.sum(point_cloud>bound[0] ,axis=-1)==3 * (np.sum(point_cloud<=bound[1] ,axis=-1)==3)) == 1
        self.root_node = octree_node(bound, 1)
        if np.sum(mask_points)>0:
            p_idx, = np.where(mask_points ==1)

            self.root_node.add_points(p_idx.tolist())
        else:
            assert False, print("no points in this bound")

        self.idx = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
        self.construct(self.root_node, self.resolution)

    def construct(self, current_node, resolution):
        if current_node.resolution == resolution:
            return
        else:

            new_resolution = current_node.resolution * 2
            num_node = 8
            new_size = (current_node.bound[1] - current_node.bound[0]) / 2

            idx = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
            xyz = np.arange(3)
            cornel_bound = np.stack([current_node.bound[0], current_node.bound[0] + new_size])

            for i in range(num_node):
                new_bound = [cornel_bound[idx[i], xyz], cornel_bound[idx[i], xyz] + new_size]
                new_node = octree_node(new_bound, new_resolution)

                if len(current_node.points_idx)>0:
                    p_idx = np.stack(current_node.points_idx)

                    point_cloud = self.point_cloud[p_idx]

                    mask_points = (np.sum(point_cloud>new_bound[0] ,axis=-1)==3 * (np.sum(point_cloud<=new_bound[1] ,axis=-1)==3))
                    
                    if np.sum(mask_points)>0:
                        valid_idx, = np.where(mask_points ==1 )
                        new_node.add_points(p_idx[valid_idx].tolist())
                        current_node.add_occ_leaves(new_node)
                    else:
                        current_node.add_unocc_leaves(new_node)
                else:
                    current_node.add_unocc_leaves(new_node)

                self.construct(new_node, resolution)



    def query_point(self, current_node, point, resolution):
        
        if np.sum((point > current_node.bound[0]) * (point <= current_node.bound[1])) ==3 and current_node.resolution == resolution:

            return current_node
        else:

            for i in range(len(current_node.occ_node)):
                if np.sum((point > current_node.occ_node[i].bound[0]) * (point <= current_node.occ_node[i].bound[1])) ==3:      
                    query_node = self.query_point(current_node.occ_node[i], point, resolution)
                    return query_node
                
            for i in range(len(current_node.unocc_node)):
                if np.sum((point > current_node.unocc_node[i].bound[0]) * (point <= current_node.unocc_node[i].bound[1])) ==3:      
                    query_node = self.query_point(current_node.unocc_node[i], point, resolution)
                    return query_node

    def random_add(self, current_node, num_points, resolution, bound_range):
        if current_node.resolution != resolution:
            for j in range(len(current_node.unocc_node)):
                self.random_add(current_node.unocc_node[j],num_points,resolution,bound_range)

            for i in range(len(current_node.occ_node)):
                self.random_add(current_node.occ_node[i],num_points,resolution,bound_range)

        else:

            if len(current_node.points_idx) == 0 and np.sum(current_node.bound[0]>=bound_range[0])==3 and np.sum(current_node.bound[1]<=bound_range[1])==3:
                bound = current_node.bound
                added_points = np.random.random((num_points,3)) * np.mean(bound[1] - bound[0]) + bound[0]
                start_point = self.point_cloud.shape[0]
                self.point_cloud = np.concatenate((self.point_cloud, added_points), axis=0)
                stop_point = self.point_cloud.shape[0]
                current_node.add_points(np.arange(start_point,stop_point).tolist())
            return

    def random_delete(self, start_idx, num_points):
            points_all = (self.point_cloud.shape[0])
            select_idx = np.random.randint(start_idx, points_all, size=(num_points,))

            return np.concatenate((self.point_cloud[:start_idx],self.point_cloud[select_idx]),axis=0)


    def add_points(self, point_cloud, resolution):
        new_point_cloud = []
        for i in range(len(point_cloud)):
            current_point = point_cloud[i]
            current_node = self.query_point(self.root_node, current_point, resolution=resolution)
            if current_node == None:

                new_point_cloud.append(point_cloud[i])
            else:
                if len(current_node.points_idx) == 0:
                    new_point_cloud.append(point_cloud[i])
        new_point_cloud = np.stack(new_point_cloud)
        new_point_cloud = np.concatenate((self.point_cloud, new_point_cloud), axis=0)
        
        return new_point_cloud

        
def compute_bound(xyz, scale_factor = 1.3):
    lower_bound = np.min(xyz,axis=0)
    higher_bound = np.max(xyz,axis=0)
    size = higher_bound - lower_bound
    max_size = np.max(size)
    box_bound = [lower_bound, lower_bound+max_size]
    ct_bound = [lower_bound, higher_bound]
    return box_bound, ct_bound




def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort


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

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    render_cameras: list
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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
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

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # print(width)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
from einops import einsum, rearrange, reduce, repeat
def get_fov(intrinsics):
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return fov_x.numpy(), fov_y.numpy()

def get_projection_matrix(
    near,
    far,
    fov_x,
    fov_y,
):
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


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


def readColmapSceneInfo(path, images, dataset, eval, rand_pcd, mvs_pcd, sparse_pcd = False, dense_pcd = False, add_rand = False, if_prune = False,  llffhold=8, N_sparse=-1, resolution=1, tau = 1000, mvs_initial = 0.85, render_path= False):

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    reading_dir = "images" if images == None else images

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    if eval:
        print("Dataset Type: ", dataset)
        if dataset == "LLFF":
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            eval_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            test_cam_infos = eval_cam_infos
            if N_sparse > 0:
                idx = list(range(len(train_cam_infos)))
                idx_train = np.linspace(0, len(train_cam_infos) - 1, N_sparse)
                idx_train = [round(i) for i in idx_train]
                idx_test = [i for i in idx if i not in idx_train] 
                train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_train]
                test_cam_infos = eval_cam_infos
        else:
            raise NotImplementedError
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        eval_cam_infos = []


    print('train', [info.image_path for info in train_cam_infos])
    print('eval', [info.image_path for info in eval_cam_infos])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if rand_pcd and mvs_pcd:
        print("[warning] Both --rand_pcd and --mvs_pcd are detected, use --mvs_pcd.")
        rand_pcd = False

    if rand_pcd:
        print('Init random point cloud.')
        ply_path = os.path.join(path, "sparse/0/points3D_random.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)



        if dataset == "LLFF":
            pcd_shape = (topk_(xyz, 1, 0)[-1] + topk_(-xyz, 1, 0)[-1])
            num_pts = int(pcd_shape.max() * 50)
            xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 20, 0)[-1]
        print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        print(xyz.shape)
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        print("using Random point cloud!")
    elif mvs_pcd:
        ply_path = os.path.join(path, "{}_views/dense/fused.ply".format(N_sparse))
        assert os.path.exists(ply_path)
        pcd = fetchPly(ply_path)

        print("using MVS for initial point cloud!")
    elif sparse_pcd:
        print('Init sparse matched point cloud.')
        ply_path = os.path.join(path, f"sparse/0/points3D_sparse_{N_sparse}.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")



        try:
            pcd = fetchPly(ply_path)
            print("fetch points from:"+ply_path)
        except:
            point_, color_ = get_sparse_correspondences(train_cam_infos,resolution, tau=tau)
            num_pts = point_.shape[0]


            pcd = BasicPointCloud(points=point_, colors=color_, normals=np.zeros((num_pts, 3)))

            storePly(ply_path, point_, color_ * 255)
            print("using sparse matcher for initial point cloud!")


    elif dense_pcd:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
        print("using COLMAP for initial point cloud!")        


    else:
        if N_sparse>0:
            ply_path = os.path.join(path, "{}_views/triangulated/points3D.ply".format(N_sparse))
            bin_path = os.path.join(path, "{}_views/triangulated/points3D.bin".format(N_sparse))
            txt_path = os.path.join(path, "{}_views/triangulated/points3D.txt".format(N_sparse))

            if not os.path.exists(ply_path):
                print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
                try:
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                except:
                    xyz, rgb, _ = read_points3D_text(txt_path)
                storePly(ply_path, xyz, rgb)
            try:
                pcd = fetchPly(ply_path)
            except:
                pcd = None


            print("using COLMAP for initial point cloud!")

        else:
            ply_path = os.path.join(path, "sparse/0/points3D.ply")
            bin_path = os.path.join(path, "sparse/0/points3D.bin")
            txt_path = os.path.join(path, "sparse/0/points3D.txt")
            if not os.path.exists(ply_path):
                print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
                try:
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                except:
                    xyz, rgb, _ = read_points3D_text(txt_path)
                storePly(ply_path, xyz, rgb)
            try:
                pcd = fetchPly(ply_path)
            except:
                pcd = None
            print("using COLMAP for initial point cloud!")

    if add_rand:
            ## add random according to sparse points range
            ply_path_l = os.path.join(path, "sparse/0/points3D.ply")
            bin_path_l = os.path.join(path, "sparse/0/points3D.bin")
            txt_path_l = os.path.join(path, "sparse/0/points3D.txt")
            if not os.path.exists(ply_path_l):
                print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
                try:
                    xyz_l, rgb_l, _ = read_points3D_binary(bin_path_l)
                except:
                    xyz_l, rgb_l, _ = read_points3D_text(txt_path_l)
                storePly(ply_path_l, xyz_l, rgb_l)
            try:
                pcd_l = fetchPly(ply_path_l)
            except:
                pcd_l = None
            print("using COLMAP for random points range!")


            # # ## random init in blank place
            point_ = pcd.points
            color_ = pcd.colors
            try:
                point_ = point_.numpy()
                color_ = color_.numpy()
            except:
                point_ = point_
                color_ = color_
                
            if dataset == "DTU":
                pcd_shape = (topk_(pcd_l.points, 100, 0)[-1] + topk_(-pcd_l.points, 100, 0)[-1])
                num_pts = 10_00
                xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-pcd_l.points, 100, 0)[-1] # - 0.15 * pcd_shape
            else:
                num_pts = 1000
                print("num random points:"+str(num_pts))
                xyz = np.random.random((num_pts, 3)) * (np.max(pcd_l.points,axis=0)-np.min(pcd_l.points,axis=0)) * 1.3 + np.min(pcd_l.points,axis=0) - (np.max(pcd_l.points,axis=0)-np.min(pcd_l.points,axis=0)) * 0.15# 1.3


            ### added without influence
            box_bound, ct_bound = compute_bound(xyz)
            oct_resolution = 32
            oct = octree(point_, box_bound, resolution=oct_resolution)  
            num_ori = len(oct.point_cloud)  
            point_ = oct.add_points(xyz,resolution=oct_resolution)

            num_added = len(point_) - num_ori

            shs = np.random.random((num_added, 3)) / 255.0

            color_ = np.concatenate((color_, SH2RGB(shs)),axis=0)
            num_pts = point_.shape[0]

            pcd = BasicPointCloud(points=point_, colors=color_, normals=np.zeros((num_pts, 3)))

            ply_path = os.path.join(path, "sparse/0/points3D_random.ply")
            storePly(ply_path, point_, color_ * 255)


    if if_prune:
        # # ## random init in blank place
        point_ = pcd.points
        color_ = pcd.colors
        normal_ = pcd.normals
        try:
            point_ = point_.numpy()
            color_ = color_.numpy()
            normal_ = normal_.numpy()
        except:
            point_ = point_
            color_ = color_
            normal_ = normal_
        ini_num = len(point_)

        prune_mask = mvs_prune(train_cam_infos,point_,res = resolution, prune_thr=mvs_initial)

        keep_mask = torch.logical_not(torch.tensor(prune_mask))
        point_ = point_[keep_mask]
        color_ = color_[keep_mask]
        normal_ = normal_[keep_mask]
        remain_num = len(point_)

        pcd = BasicPointCloud(points=point_, colors=color_, normals=normal_)

        print("original number of points:"+str(ini_num))
        print("prune points:"+str(ini_num-remain_num))
        ply_path = os.path.join(path, "sparse/0/points3D_prune.ply")
        storePly(ply_path, point_, color_ * 255)


    if render_path:

        factor = 8
        with open(os.path.join(path, 'poses_bounds.npy'),
                            'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        bounds = poses_arr[:, -2:]

        # Pull out focal length before processing poses.
        focal = poses[0, -1, -1] / factor

        # Correct rotation matrix ordering (and drop 5th column of poses).
        fix_rotation = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
                                dtype=np.float32)

        poses = poses[:, :3, :4] @ fix_rotation

        # Rescale according to a default bd factor.
        scale = 1. / (bounds.min() * .75)
        poses[:, :3, 3] *= scale
        bounds *= scale

        width = int(train_cam_infos[0].width/factor)
        height = int(train_cam_infos[0].height/factor)
        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)
        # Center and scale poses.
        camtoworlds = poses
        render_cam_infos = []
        render_poses = generate_spiral_path(camtoworlds, bounds, fix_rotation, n_frames=120)

        for i, render_pose in enumerate(render_poses):
            # get the world-to-camera transform and set R, T
            pose = np.eye(4, dtype=np.float32)
            pose[:3] = render_pose
            w2c = np.linalg.inv(pose)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=train_cam_infos[0].image,
                                    image_path=None, image_name=f"{i}", width=width, height=height)
            render_cam_infos.append(cam_info)
        
    
    else:
        render_cam_infos = []
        

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info



def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
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

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, rand_pcd, sparse_pcd = False, add_rand = False, if_prune = False, llffhold=8, N_sparse=-1, extension=".png" , resolution = -1, tau=0.0001, mvs_initial=0.85):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if eval:
        if N_sparse > 0:
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in [2, 16, 26, 55, 73, 76, 86, 93]]
            train_cam_infos = train_cam_infos[:N_sparse]
        eval_cam_infos = [c for idx, c in enumerate(test_cam_infos) if idx % llffhold == 0]
        # test_cam_infos = test_cam_infos
        if N_sparse > 0:
            test_cam_infos = eval_cam_infos
        else:
            test_cam_infos = test_cam_infos

    else:
        test_cam_infos = []
        eval_cam_infos = []


    print('train', [info.image_path for info in train_cam_infos])
    print('eval', [info.image_path for info in eval_cam_infos])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if rand_pcd:
        print('Init random point cloud.')
    if rand_pcd: # or not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2 - 1
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif sparse_pcd:
        print('Init sparse matched point cloud.')
        ply_path = os.path.join(path, f"points3D_sparse_{N_sparse}.ply")
        bin_path = os.path.join(path, "points3D.bin")
        txt_path = os.path.join(path, "points3D.txt")
        
        point_, color_ = get_sparse_correspondences(train_cam_infos,resolution, tau=tau)
        num_pts = point_.shape[0]
        pcd = BasicPointCloud(points=point_, colors=color_, normals=np.zeros((num_pts, 3)))



        storePly(ply_path, point_, color_ * 255)
        print("using sparse matcher for initial point cloud!")


    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    if add_rand:
        # # # ## random init in blank place
        point_ = pcd.points
        color_ = pcd.colors
        try:
            point_ = point_.numpy()
            color_ = color_.numpy()
        except:
            point_ = point_
            color_ = color_
        num_pts = 500 #  500
        xyz = np.random.random((num_pts, 3)) * 2 - 1
        shs = np.random.random((num_pts, 3)) / 255.0
        point_ = np.concatenate((point_,xyz),axis=0)
        color_ = np.concatenate((color_,SH2RGB(shs)),axis=0)
        num_pts = point_.shape[0]
        print("num all points:"+str(num_pts))

        pcd = BasicPointCloud(points=point_, colors=color_, normals=np.zeros((num_pts, 3)))

        ply_path = os.path.join(path, "points3D_random.ply")
        storePly(ply_path, point_, color_ * 255)

    if if_prune:
            # # ## random init in blank place
            point_ = pcd.points
            color_ = pcd.colors
            normal_ = pcd.normals
            try:
                point_ = point_.numpy()
                color_ = color_.numpy()
                normal_ = normal_.numpy()
            except:
                point_ = point_
                color_ = color_
                normal_ = normal_
            ini_num = len(point_)
            prune_mask = mvs_prune(train_cam_infos,point_,res = resolution,prune_thr=mvs_initial)
            keep_mask = torch.logical_not(torch.tensor(prune_mask))
            # keep_mask = prune_mask

            point_ = point_[keep_mask]
            color_ = color_[keep_mask]
            normal_ = normal_[keep_mask]
            remain_num = len(point_)


            pcd = BasicPointCloud(points=point_, colors=color_, normals=normal_)

            print("original number of points:"+str(ini_num))
            print("prune points:"+str(ini_num-remain_num))

    render_cam_infos = []

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras = render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
}
