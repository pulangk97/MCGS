num_views=8
base_path=/media/xyr/data22/datasets/datasets/nerf_synthetic
bash scripts/train_eval_blender.sh $base_path chair $num_views sa
bash scripts/train_eval_blender.sh $base_path drums $num_views sa
bash scripts/train_eval_blender.sh $base_path ficus $num_views sa
bash scripts/train_eval_blender.sh $base_path hotdog $num_views sa
bash scripts/train_eval_blender.sh $base_path lego $num_views sa
bash scripts/train_eval_blender.sh $base_path materials $num_views sa
bash scripts/train_eval_blender.sh $base_path mic $num_views sa
bash scripts/train_eval_blender.sh $base_path ship $num_views sa