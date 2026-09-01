# FedRetrain on Tiny ImageNet

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASETS=("tiny_imagenet")

# create logs directory if not exists
mkdir -p logs/tiny_imagenet

# echo "Running FedRetrain on $DATASET"
for DATASET in "${DATASETS[@]}"; do
    # get enumerated index for dataset
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr 0.01 --unlearn_method FedRetrain --global_epochs 200 --local_epochs 2 --refine_epochs 0 --pretrain_model FedAvg_${DATASET}_warmup --save FedRetrain_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/${DATASET}/FedRetrain_${SEED}.log" 2>&1 &
        if (( ($SEED + 1) % 3 == 0 )); then
            wait
        fi
    done
done