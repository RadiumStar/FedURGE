# Federated Memory Pruning (FedMemPrune) for SVHN
 
SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASET="svhn"
GLOBAL_EPOCHS=50
CUDA_GAP=3
LR=0.0001

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

for SEED in "${SEEDS[@]}"; do   
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --optimizer Adam --global_epochs ${GLOBAL_EPOCHS} --local_epochs 2 --unlearn_method FedMemPrune --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedMemPrune_${DATASET}_${SEED}" --unlearn_acc_threshold 0.1 --cuda $(( $SEED % 3 )) > "logs/${DATASET}/fedmemprune_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi 
done 