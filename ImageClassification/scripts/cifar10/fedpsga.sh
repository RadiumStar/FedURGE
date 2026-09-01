# Federated Unlearning with Projected Gradient Ascent for CIFAR-10

CUDA_GAPS=(6 3) 

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASET="cifar10"

UNLEARN_EPOCHS=8
REFINE_EPOCHS=20
LR=0.003  
CUDA_GAP=3

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

# run the unlearning process with FedPSGA method
echo "Running FedPSGA on $DATASET (unlearn_epochs=$UNLEARN_EPOCHS, refine_epochs=$REFINE_EPOCHS)" 

for SEED in "${SEEDS[@]}"; do
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedPSGA --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs $REFINE_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedPSGA_${DATASET}_${SEED}" --unlearn_acc_threshold 0.1 --cuda $(($SEED % 3)) > "logs/${DATASET}/fedpsga_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi
done  