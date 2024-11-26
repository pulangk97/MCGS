#!/bin/bash

# Base path and number of views
num_views=3
base_path=/media/xyr/data22/datasets/datasets/nerf_llff_data


datasets=("fern" "flower" "horns" "orchids" "trex" "room" "leaves" "fortress")

for dataset in "${datasets[@]}"; do
    bash scripts/train_eval_llff.sh "$base_path" "$dataset" "$num_views" "mcgs"
done
 