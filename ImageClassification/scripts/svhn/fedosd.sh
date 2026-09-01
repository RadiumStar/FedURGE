# FedOSD (Federated Unlearning with Orthogonal Steepest Descent) for image classification
# Originally a client-removal algorithm (AAAI 2025), migrated here to the sample-removal task.
#
# Per-dataset tuned hyperparameters (seed-0 tuned):
#   dataset        lr     ft_lr  unlearn_epochs  refine_epochs  batch_size  orthogonal
#   cifar10        0.01   0.01   20              15             128         on
#   svhn           0.03   0.01   45              15             128         off
#   cifar100       0.2    0.01   80              15             128         off
#   tiny_imagenet  0.03   0.01   10              10             64          off
#
# `orthogonal` enables the Orthogonal Steepest Descent projection (FedOSD core). For
# high-class-count datasets (cifar100, tiny_imagenet) the projection was found to slow
# forgetting and lower accuracy, so it is switched off via FEDOSD_ORTHOGONAL=0. svhn uses
# more unlearning rounds (45) with projection off to reliably suppress the backdoor across
# all seeds (some seeds have a strong/stubborn backdoor). cifar100 uses a higher lr (0.2,
# with a small accuracy cost) to drive the backdoor much lower within the fixed rounds.

SEEDS=(0 1 2 3 4 5 6 7 8 9)

DATASET="svhn"
LR=0.03
FT_LR=0.01
UNLEARN_EPOCHS=45
REFINE_EPOCHS=15
BATCH_SIZE=128
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