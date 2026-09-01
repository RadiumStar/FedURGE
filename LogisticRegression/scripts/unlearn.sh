mkdir -p logs/methods

UNLEARN_ALGS=("fedgb" "fedgd" "fedcup")
PARTITION=iid
ALPHA=0.5
SEEDS=(0 1 2 3 4)
# DELTAS=(0.9 0.99 0.999 0.9999)
DELTA=0.9999

for SEED in "${SEEDS[@]}"; do 
    for UNLEARN_ALG in "${UNLEARN_ALGS[@]}"; do 
        python -u unlearn.py --seed $SEED  --unlearn_alg $UNLEARN_ALG --global_epochs 5001 --partition $PARTITION --alpha $ALPHA --delta $DELTA > logs/methods/unlearn_${UNLEARN_ALG}_seed${SEED}.log 2>&1 & 
        wait
    done 
    echo "Waiting for all unlearning processes to finish for seed $SEED..."
    wait 
done 