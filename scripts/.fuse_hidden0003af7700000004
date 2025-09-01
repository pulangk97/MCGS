num_views=12
base_path=./data/360

scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")

for scene in "${scenes[@]}"; do
    bash scripts/train_eval_360_12.sh $base_path $scene $num_views mcgs
done