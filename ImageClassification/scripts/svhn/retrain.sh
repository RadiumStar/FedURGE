# FedRetrain for SVHN

SEEDS=(0 1 2 3 4 5 6 7 8 9)
SEEDS=(6 7 8 9)
DATASET="svhn"

# create log directory if not exist
mkdir -p "logs/${DATASET}"

for SEED in "${SEEDS[@]}"; do
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --unlearn_method FedRetrain --pretrain_model "FedAvg_${DATASET}_${SEED}" --save FedRetrain_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/${DATASET}/FedRetrain_${SEED}.log" 2>&1 &
done