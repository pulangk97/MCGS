data_dir=$1
scene=$2
num_views=$3
method=$4
model_path=./output/$method/dtu_$num_views/$scene
python train.py -s "$data_dir/$scene" --eval --n_sparse $num_views -m "$model_path" --resolution 4 --dataset "DTU"
python render.py -s "$data_dir/$scene" -m "$model_path" --iteration 10000 --render_depth
python metrics.py -m "$model_path"