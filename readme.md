## [MCGS: Multiview Consistency Enhancement for Sparse-View 3D Gaussian Radiance Fields](https://arxiv.org/abs/2410.11394)

## Overview
![method](./assets/method.png)

## Installation
Install the following dependency
```
conda env create -f environment.yml  

cd submodules
git clone git@github.com:ashawkey/diff-gaussian-rasterization.git --recursive
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git

cd ..
pip install ./submodules/diff-gaussian-rasterization  
pip install ./submodules/simple-knn  
pip install ./LightGlue
```

## Dataset
Download Blender Dataset from [Blender](https://github.com/bmild/nerf)
Download LLFF Dataset from [LLFF](https://github.com/Fyusion/LLFF)
Download MipNeRF360 Dataset from [360](https://jonbarron.info/mipnerf360)
## Usage
### LLFF Dataset
```
bash ./scripts/llff.sh
```
### Blender Dataset
```
bash ./scripts/blender.sh
```
 
### MipNeRF360 Dataset
```
bash ./scripts/360_12.sh
```
### Citation
```
@article{xiao2024mcgs,
  title={MCGS: Multiview Consistency Enhancement for Sparse-View 3D Gaussian Radiance Fields},
  author={Xiao, Yuru and Zhai, Deming and Zhao, Wenbo and Jiang, Kui and Jiang, Junjun and Liu, Xianming},
  journal={arXiv preprint arXiv:2410.11394},
  year={2024}
}
```
 
