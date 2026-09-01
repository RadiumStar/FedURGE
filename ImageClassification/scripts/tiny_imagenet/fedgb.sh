# Federated Unlearning with Gradient Balancing for Tiny ImageNet

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASET="tiny_imagenet"
UNLEARN_EPOCHS=15
REFINE_EPOCHS=0
LR=0.005
FT_LR=0.0001
CUDA_GAP=3

for SEED in "${SEEDS[@]}"; do
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedGB method
    echo "Running FedGB on $DATASET (unlearn_epochs=$UNLEARN_EPOCHS, refine_epochs=$REFINE_EPOCHS)" 
    
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr $FT_LR --unlearn_method FedGB --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs $REFINE_EPOCHS --local_epochs 2 --batch_size 64 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedGB_${DATASET}_${SEED}" --unlearn_acc_threshold 0.01 --cuda $(($SEED % 3)) > "logs/${DATASET}/fedgb_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi
done 