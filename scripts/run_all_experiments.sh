#!/bin/bash
# CS-BAPR paper experiments: multi-seed + ablation
# Run on server in background. Each method×seed takes ~4.5h on single-thread CPU.
# Recommend GNU parallel or nohup batches.
#
# Total: 4 methods × 3 seeds = 12 runs × 4.5h = ~54 CPU-hours
# (can parallelize if server has multiple cores)

set -e
LOGDIR=${LOGDIR:-/tmp/csbapr_exp}
EPISODES=${EPISODES:-500}
mkdir -p "$LOGDIR"

METHODS=("csbapr" "bapr" "csbapr-kan" "csbapr-no-nau")
SEEDS=(0 1 2)

cd "$(dirname "$0")/.."

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log="${LOGDIR}/${method}_seed${seed}.log"
        if [ -s "$log" ] && grep -q "Total time" "$log"; then
            echo "SKIP $method seed=$seed (already complete)"
            continue
        fi
        echo "START $method seed=$seed → $log"
        python -u scripts/test_multiline_convergence.py \
            --episodes "$EPISODES" --seed "$seed" --method "$method" \
            > "$log" 2>&1
    done
done

echo "ALL DONE. Collect results with scripts/collect_results.py"
