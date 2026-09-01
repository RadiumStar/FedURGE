# Federated Unlearning Compensation (FedCUP, layer-wise) for image classification

SEEDS=(0 1 2 3 4 5 6 7 8 9)

DATASETS=("cifar10" "svhn" "cifar100" "tiny_imagenet")
UNLEARN_EPOCHS_ARR=(50 50 50 40)
LOCAL_EPOCHS_ARR=(2 2 2 4)
CUDA_AVAILABLE=(0 1 2)
CUDA_GAPS=(3 3 3 3)
BATCH_SIZES=(128 128 128 64)

LRS_ARR=(0.01 0.01 0.01 0.015)
DELTAS_ARR=(0.9999 0.9999 0.9999 0.9999)

# FedCUP constricts the unlearning gradient per layer (layerwise) by default; 
# set LAYERWISE=0 to use a single global constriction radius instead. 
LAYERWISE=1

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}" 

    # only run tiny_imagenet
    if [[ "$DATASET" != "svhn" ]]; then
        continue
    fi

    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    DELTA="${DELTAS_ARR[$i]}"
    BATCH_SIZE="${BATCH_SIZES[$i]}"
    CUDA_GAP="${CUDA_GAPS[$i]}"

    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with the FedCUP method 
    echo "Running FedCUP on $DATASET (lr=$LR, delta=$DELTA, layerwise=$LAYERWISE, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)"
    for SEED in "${SEEDS[@]}"; do
        CUDA="${CUDA_AVAILABLE[$SEED % ${#CUDA_AVAILABLE[@]}]}"
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR \
            --unlearn_method FedCUP --delta $DELTA --layerwise 1 --batch_size $BATCH_SIZE \
            --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 \
            --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedCUP_${DATASET}_${SEED}" \
            --cuda $CUDA > "logs/${DATASET}/FedCUP_${DATASET}_${SEED}.log" 2>&1 &

        if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
            wait
        fi
    done
    wait 
done
