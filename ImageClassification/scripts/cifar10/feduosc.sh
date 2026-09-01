# Federated Unlearning with Oriented Saliency Compression for CIFAR-10

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASETS=("cifar10")

UNLEARN_EPOCHS_ARR=(10)
REFINE_EPOCHS_ARR=(2)
LRS_ARR=(0.005)
FT_LRS_ARR=(0.0001)

CUDA_GAPS=(6)

for i in "${!DATASETS[@]}"; do 
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    REFINE_EPOCHS="${REFINE_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    FT_LR="${FT_LRS_ARR[$i]}"

    CUDA_GAP="${CUDA_GAPS[$i]}"
    
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedUOSC method
    echo "Running FedUOSC on $DATASET (unlearn_epochs=$UNLEARN_EPOCHS, refine_epochs=$REFINE_EPOCHS)" 
    
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr $FT_LR --unlearn_method FedUOSC --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs $REFINE_EPOCHS --local_epochs 2 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedUOSC_${DATASET}_${SEED}" --unlearn_acc_threshold 0.1 --cuda $(($SEED % 3)) > "logs/${DATASET}/feduosc_${DATASET}_${SEED}.log" 2>&1 &

        if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
            wait
        fi
    done 
    wait
done 