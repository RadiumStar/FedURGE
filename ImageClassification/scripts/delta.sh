# Federated Unlearning with Residual Gradient Estimation (FedURGE, layer-wise) for image classification

SEEDS=(0 1 2 3 4 5 6 7 8 9)

DATASETS=("cifar10" "cifar100")
UNLEARN_EPOCHS=50
LOCAL_EPOCHS=2
CUDA_AVAILABLE=(0 1 2)
CUDA_GAPS=(6 3) 

LR=0.01
DELTAS_ARR=(0.7 0.75 0.8 0.85 0.95 0.9999)
LAYERWISE=1

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}" 
    CUDA_GAP="${CUDA_GAPS[$i]}"

    # create log directory if it doesn't exist
    mkdir -p "logs/deltas"

    # run the unlearning process with the FedURGE method 
    for DELTA in "${DELTAS_ARR[@]}"; do
        echo "Running FedURGE on $DATASET (lr=$LR, delta=$DELTA, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)"
        for SEED in "${SEEDS[@]}"; do
            CUDA="${CUDA_AVAILABLE[$SEED % ${#CUDA_AVAILABLE[@]}]}"
            python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR \
                --unlearn_method FedURGE --delta $DELTA --layerwise 1 \
                --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 \
                --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedURGE_${DATASET}_${SEED}" \
                --cuda $CUDA > "logs/deltas/delta${DELTA}_${DATASET}_${SEED}.log" 2>&1 &

            if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
                wait
            fi
        done
        wait 
    done
done
