mkdir -p logs/pretrain

PARTITION=iid
ALPHA=0.5
SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do 
    echo "Running pretrain on seed $SEED..."
    python -u pretrain.py --seed $SEED --partition $PARTITION --alpha $ALPHA > logs/pretrain/pretrain_seed${SEED}.log 2>&1 
    wait
    echo "Finished pretrain for seed $SEED."
done 
