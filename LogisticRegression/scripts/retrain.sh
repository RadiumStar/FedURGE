mkdir -p logs/retrain

PARTITION=iid
ALPHA=0.5
SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do 
    echo "Running retrain on seed $SEED..."
    python -u retrain.py --seed $SEED --partition $PARTITION --alpha $ALPHA > logs/retrain/retrain_seed${SEED}.log 2>&1 
    wait
    echo "Finished retrain for seed $SEED."
done 
