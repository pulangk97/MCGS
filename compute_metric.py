import numpy as np
import os
import json


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



source_path = "/media/xyr/data11/code/3DGS/gaussian-splatting-wdepth/gaussian-splatting-diff-depth/MCGS/output/sa/blender_8"
ckp_name = "ours_10000"
compute_ave_metrics(source_path, ckp_name)

    

