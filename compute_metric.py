import numpy as np
import os
import json
# source_path = "/data_b/xiaoyuru/xyr/3DGS/3DGS_base/output/gs/llff_3_fssetting_maskabl/mask_0.05"
# ckp_name = "ours_10000"

def compute_ave_metrics(source_path, ckp_name):
    target_path = source_path+"/results.json"


    scenes = os.listdir(source_path)
    psnr = []
    ssim = []
    lpips = []
    for scene in scenes:
        
        result_path = os.path.join(source_path,scene) + "/results.json"
        if not os.path.exists(result_path):
            continue
        with open(result_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            psnr.append(data[ckp_name]["PSNR"])
            ssim.append(data[ckp_name]["SSIM"])
            lpips.append(data[ckp_name]["LPIPS"])
            # print(data[ckp_name])
    ave_psnr = sum(psnr)/len(psnr)     
    ave_ssim = sum(ssim)/len(ssim)    
    ave_lpips = sum(lpips)/len(lpips)   

    data = {ckp_name:{
        "PSNR": ave_psnr,
        "SSIM": ave_ssim,
        "LPIPS": ave_lpips,
    }}
    with open(target_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# source_path = "/media/xyr/data11/code/3DGS/gaussian-splatting-wdepth/gaussian-splatting-diff-depth/gaussian-splatting/output/sa/llff_3"
source_path = "/media/xyr/data11/code/3DGS/gaussian-splatting-wdepth/gaussian-splatting-diff-depth/MCGS/output/mcgs/blender_4"
ckp_name = "ours_10000"
compute_ave_metrics(source_path, ckp_name)
# for i in range(5,105,5):
#     abl_dir = f"mask_{float(i)/100:.2f}"
#     compute_ave_metrics(source_path+"/"+abl_dir, ckp_name)
    

