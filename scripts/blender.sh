#!/bin/bash

# Base path and number of views
num_views=8
base_path=/media/xyr/data22/datasets/datasets/nerf_synthetic

 
datasets=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for dataset in "${datasets[@]}"; do
    bash scripts/train_eval_blender.sh "$base_path" "$dataset" "$num_views" "mcgs"
done
 