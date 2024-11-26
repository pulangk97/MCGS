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
 
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False
##
import torch
import torch.nn.functional as F
def get_img_feature(img, model, if_cont_rgb=False, layeridx = [2,6,40,84]): #[2,6,40,84]
    def get_featuremap(layer_output, image_shape, idx=[2,6,40,84]):
        featuremap = torch.tensor([])
        for i in idx:
            out = layer_output[i]
            for key, value in out.items():
                scale_factor = (image_shape[0]/value.shape[-2],image_shape[1]/value.shape[-1])
                output_tensor = F.interpolate(value, scale_factor=scale_factor, mode='bilinear', align_corners=True)

                featuremap = torch.concat((featuremap, output_tensor), dim=1)

        return featuremap

    def forward_hook(module, input, output):

        class_name = module.__class__.__name__
        data = { class_name: output

        }
        layer_outputs.append(data)

    layer_outputs = []

    for name, layer in model.named_modules():
        layer.register_forward_hook(forward_hook)

    model = model
    output = model(img)
    shape = img.shape[-2:]
    featuremap = get_featuremap(layer_output=layer_outputs, image_shape=shape, idx=layeridx)   
    if if_cont_rgb == True:
        
        featuremap = torch.concat((img[None,...],featuremap),dim=1) 
    featuremap = featuremap/torch.norm(featuremap,dim=1)
    return featuremap.squeeze()


def loadCam(args, id, cam_info, resolution_scale, feature_extract_model):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None
    
    ############# get image feature from gt image by DINO ###############
    # resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    if feature_extract_model!=None:
        gt_feature = get_img_feature(gt_image[None,...], model = feature_extract_model)
    else:
        gt_feature = None
    #############                                         ###############

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    #############     return Camera with gt feature   ###############
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, image_feature = gt_feature, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, test=False):
    camera_list = []
    ############# initialize model ###############
    if test==False:
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    else:
        resnet50 = None

    
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, resnet50))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
