data_dir=$1
scene=$2
num_views=$3
method=$4
model_path=./output/$method/blender_$num_views/$scene
python train.py -s "$data_dir/$scene" --eval --n_sparse $num_views -m "$model_path" --resolution 2 \
        --white_background --densify_grad_threshold 0.0003 --mvs_prune_iterval 2000 --tv_weight 0.01 --tv_start 3000 --occ_reg_weight 0.05 \
        --start_point "[0,64,128,256]" --mvs_prune_thresholds "[0.6, 0.65, 0.7, 0.8]" --sparse_pcd --add_rand --train_mvs_prune --if_TV --port 6789
python render.py -s "$data_dir/$scene" -m "$model_path" --iteration 10000
python metrics.py -m "$model_path"