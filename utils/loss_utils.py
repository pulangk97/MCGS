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
 
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

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

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

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

