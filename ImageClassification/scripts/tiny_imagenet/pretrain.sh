# FedAvg on Tiny ImageNet

# 0. create logs directory if not exists
mkdir -p logs/pretrain

# 1. Since Tiny ImageNet is a larger and more complex dataset, we first pretrain a global model using FedAvg for 50 epochs (cosine annealing) as the warmup model on a **CLEAN** dataset. You can open the cosine annealing scheduler in `core/fedavg.py`, line 31 - 36. The pretrained model will be saved as "FedAvg_tiny_imagenet_warmup".

# python -u unlearning.py --seed 0 --config "config/tiny_imagenet.yml" --lr 0.05 --global_epochs 50 --local_epochs 2 --unlearn_perc 0.0 --save FedAvg_tiny_imagenet_warmup --cuda 0 > "logs/pretrain/pretrain_tiny_imagenet_warmup.log" 2>&1 &

# 2. Then we use the pretrained model to continue training for 200 epochs on the backdoor dataset, and save the final model as "FedAvg_tiny_imagenet_{SEED}" for later unlearning.

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASETS=("tiny_imagenet")


for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do   
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr 0.01 --global_epochs 200 --local_epochs 2 --pretrain_model "FedAvg_${DATASET}_warmup" --save FedAvg_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/pretrain/pretrain_${DATASET}_${SEED}.log" 2>&1 &
    done
    if (( ($SEED + 1) % 3 == 0 )); then
        wait
    fi
done