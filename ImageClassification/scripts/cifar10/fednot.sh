# Federated Unlearning via Weight Negation for CIFAR-10

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASETS=("cifar10")

UNLEARN_EPOCHS_ARR=(150)
LOCAL_EPOCHS_ARR=(1)
LRS_ARR=(0.006)
CUDA_GAPS=(2)
CUDAS=(1)

for i in "${!DATASETS[@]}"; do  
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    CUDA_GAP="${CUDA_GAPS[$i]}"
    
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedNot method
    echo "Running FedNot on $DATASET (global_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 
    
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedNot --global_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedNot_${DATASET}_${SEED}" --unlearn_acc_threshold 0.15 --cuda "${CUDAS[$(($SEED % 1))]}" > "logs/${DATASET}/fednot_${DATASET}_${SEED}.log" 2>&1 &

        if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
            wait
        fi
    done 
    wait
done