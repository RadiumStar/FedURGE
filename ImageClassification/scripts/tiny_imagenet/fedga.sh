# Federated Unlearning with Gradient Ascent on Tiny-ImageNet

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASET="tiny_imagenet"

UNLEARN_EPOCHS=10
REFINE_EPOCHS=10
LR=0.005

CUDA_GAP=3
    
# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

# run the unlearning process with FedGA method
echo "Running FedGA on $DATASET (unlearn_epochs=$UNLEARN_EPOCHS, refine_epochs=$REFINE_EPOCHS)" 

for SEED in "${SEEDS[@]}"; do
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedGA --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs $REFINE_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedGA_${DATASET}_${SEED}" --unlearn_acc_threshold 0.1 --cuda $(($SEED % 3)) > "logs/${DATASET}/fedga_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi
done  