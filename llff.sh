num_views=3
base_path=/media/xyr/data22/datasets/datasets/nerf_llff_data
bash scripts/train_eval_llff.sh $base_path fern $num_views sa
bash scripts/train_eval_llff.sh $base_path flower $num_views sa
bash scripts/train_eval_llff.sh $base_path horns $num_views sa
bash scripts/train_eval_llff.sh $base_path orchids $num_views sa
bash scripts/train_eval_llff.sh $base_path trex $num_views sa
bash scripts/train_eval_llff.sh $base_path room $num_views sa
bash scripts/train_eval_llff.sh $base_path leaves $num_views sa
bash scripts/train_eval_llff.sh $base_path fortress $num_views sa