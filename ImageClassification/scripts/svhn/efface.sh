# Federated Random Labeling (FedRL) for SVHN

SEEDS=(0 1 2 3 4 5 6 7 8 9)

DATASET="svhn"
UNLEARN_EPOCHS=80
LOCAL_EPOCHS=1
LR=0.005
CUDA_GAP=3

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

# run the unlearning process with FedRL method
echo "Running FedRL on $DATASET (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 

for SEED in "${SEEDS[@]}"; do
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedRL --communicator EFFACE --compressor TopK --k 1 --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda $(($SEED % 3)) >> "logs/${DATASET}/EFFACE_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi
done 