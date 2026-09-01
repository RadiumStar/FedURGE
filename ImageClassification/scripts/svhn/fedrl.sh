# Federated Random Labeling (FedRL) for SVHN

SEEDS=(0 1 2 3 4 5 6 7 8 9)
DATASETS=("svhn")

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedRL method
    echo "Running FedRL on $DATASET" 
    
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr 0.005 --unlearn_method FedRL --communicator EFFACE --compressor TopK --k 1 --unlearn_epochs 80 --local_epochs 1 --refine_epochs 0 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda $(($SEED % 3)) >> "logs/${DATASET}/fedrl_${DATASET}_${SEED}.log" 2>&1 &

        if (( ($SEED + 1) % 3 == 0 )); then
            wait
        fi
    done 
    wait
done