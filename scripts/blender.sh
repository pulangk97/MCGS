#!/bin/bash

# Base path and number of views
num_views=4
base_path=./data/nerf_synthetic

 
datasets=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for dataset in "${datasets[@]}"; do
    bash scripts/train_eval_blender.sh "$base_path" "$dataset" "$num_views" "mcgs"
done
 