data_dir=$1
scene=$2
num_views=$3
method=$4
model_path=./output/$method/360_$num_views/$scene
python train.py -s "$data_dir/$scene" --eval --n_sparse $num_views -m "$model_path" --dataset "LLFF" --resolution 8 --iterations 15000 --position_lr_max_steps 15000 \
        --densify_until_iter 12000 \
        --densify_grad_threshold 0.0005 \
        --mvs_prune_end 12000 --mvs_prune_iterval 1000 \
        --tv_start 12000 --mvs_prune_threshold_initial 0.5 --tv_weight 0.0003 \
        --if_prune --sparse_pcd --add_rand --train_mvs_prune --if_TV
python render.py -s "$data_dir/$scene" -m "$model_path" --iteration 15000 --render_depth
python metrics.py -m "$model_path"
