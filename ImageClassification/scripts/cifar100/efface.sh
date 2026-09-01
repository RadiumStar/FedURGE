# Federated Random Labeling (FedRL) for CIFAR-100

SEEDS=(0 1 2 3 4 5 6 7 8 9)

# DATASETS=("cifar10" "svhn" "cifar100" "tiny_imagenet" "mnist" "fashion_mnist")  
# UNLEARN_EPOCHS_ARR=(80 80 100 100 60 60)
# LOCAL_EPOCHS_ARR=(1 1 1 2 1 1)
# CUDA_GAPS=(2 2 1 1 3 3)
# LRS_ARR=(0.001 0.005 0.005 0.005 0.005 0.005) 

DATASET="cifar100"
UNLEARN_EPOCHS=100
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