# Federated Gradient Descent (FedGD) for Tiny-ImageNet

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASETS=("tiny_imagenet")

UNLEARN_EPOCHS_ARR=(200)
LOCAL_EPOCHS_ARR=(2)
LRS_ARR=(0.001)

for i in "${!DATASETS[@]}"; do  
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedGD method
    echo "Running FedGD on $DATASET (unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 
    
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedGD --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedGD_${DATASET}_${SEED}" --cuda $(($SEED % 3)) > "logs/${DATASET}/fedgd_${DATASET}_${SEED}.log" 2>&1 &

        if (( ($SEED + 1) % 3 == 0 )); then
            wait
        fi
    done 
    wait
done
