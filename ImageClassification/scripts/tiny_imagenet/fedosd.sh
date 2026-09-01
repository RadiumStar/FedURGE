SEEDS=(0 1 2 3 4 5 6 7 8 9)

DATASET="tiny_imagenet"
LR=0.03
FT_LR=0.01
UNLEARN_EPOCHS=10
REFINE_EPOCHS=10
BATCH_SIZE=64
ORTHOGONAL=0

CUDA=0

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

echo "Running FedOSD on $DATASET (lr=$LR, ft_lr=$FT_LR, unlearn_epochs=$UNLEARN_EPOCHS, refine_epochs=$REFINE_EPOCHS, batch_size=$BATCH_SIZE, orthogonal=$ORTHOGONAL)"

for SEED in "${SEEDS[@]}"; do
    # run seeds sequentially on the single available GPU
    FEDOSD_ORTHOGONAL=$ORTHOGONAL python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" \
        --lr $LR --ft_lr $FT_LR \
        --unlearn_method FedOSD --batch_size $BATCH_SIZE \
        --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs $REFINE_EPOCHS \
        --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedOSD_${DATASET}_${SEED}" \
        --cuda $CUDA > "logs/${DATASET}/fedosd_${DATASET}_${SEED}.log" 2>&1
done