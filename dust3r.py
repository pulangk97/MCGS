from extension.dust3r.dust3r.inference import inference
from extension.dust3r.dust3r.model import AsymmetricCroCo3DStereo
from extension.dust3r.dust3r.utils.image import load_images
from extension.dust3r.dust3r.image_pairs import make_pairs
from extension.dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import numpy as np
import os


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "/media/xyr/data11/code/3DGS/gaussian-splatting-wdepth/gaussian-splatting-diff-depth/MCGS/extension/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory


    datadir = "/media/xyr/data22/datasets/datasets/llff/nerf_llff_data/fern/images_8"
    dataset = "LLFF"
    N_sparse = 3
    llffhold = 8
    list_dir_imgs = os.listdir(datadir)
    list_dir_imgs_full = []
    for dir in list_dir_imgs:
        list_dir_imgs_full.append(os.path.join(datadir, dir))
    if dataset == "LLFF":
        train_cam_infos = [c for idx, c in enumerate(list_dir_imgs_full) if idx % llffhold != 0]
        eval_cam_infos = [c for idx, c in enumerate(list_dir_imgs_full) if idx % llffhold == 0]
        if N_sparse > 0:
            idx = list(range(len(train_cam_infos)))
            idx_train = np.linspace(0, len(train_cam_infos) - 1, N_sparse)
            idx_train = [round(i) for i in idx_train]
            idx_test = [i for i in idx if i not in idx_train] 
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_train]
            test_cam_infos = eval_cam_infos


    images = load_images(list_dir_imgs_full, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # for i, c in enumerate(pairs):
    #     pairs[i]
    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    print(poses.shape)