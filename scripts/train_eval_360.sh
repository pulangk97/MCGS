data_dir=$1
scene=$2
num_views=$3
method=$4
model_path=./output/$method/360_$num_views/$scene
python train.py -s "$data_dir/$scene" --eval --n_sparse $num_views -m "$model_path" --dataset "LLFF" --resolution 8
python render.py -s "$data_dir/$scene" -m "$model_path" --iteration 10000 --render_depth
python metrics.py -m "$model_path"