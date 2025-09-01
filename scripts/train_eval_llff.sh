data_dir=$1
scene=$2
num_views=$3
method=$4
model_path=./output/$method/llff_$num_views/$scene
python train.py -s "$data_dir/$scene" --eval --n_sparse $num_views -m "$model_path" --dataset "LLFF" --resolution 8 \
                --if_prune --add_rand --train_mvs_prune --if_TV --mvs_prune_threshold_initial 0.7 --tv_weight 0.0003 --sparse_pcd
python render.py -s "$data_dir/$scene" -m "$model_path" --iteration 10000 --render_depth
python metrics.py -m "$model_path"
