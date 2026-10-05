#!/bin/bash
# Parallel version — launches all 12 runs at once (needs ≥12 CPU cores)
# Use on strong server. Each run uses ~3-5 cores for env simulation.

set -e
LOGDIR=${LOGDIR:-/tmp/csbapr_exp}
EPISODES=${EPISODES:-500}
mkdir -p "$LOGDIR"

METHODS=("csbapr" "bapr" "csbapr-kan" "csbapr-no-nau")
SEEDS=(0 1 2)

cd "$(dirname "$0")/.."

pids=()
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log="${LOGDIR}/${method}_seed${seed}.log"
        if [ -s "$log" ] && grep -q "Total time" "$log"; then
            echo "SKIP $method seed=$seed"
            continue
        fi
        echo "LAUNCH $method seed=$seed → $log"
        python -u scripts/test_multiline_convergence.py \
            --episodes "$EPISODES" --seed "$seed" --method "$method" \
            > "$log" 2>&1 &
        pids+=($!)
    done
done

echo "Launched ${#pids[@]} runs. PIDs: ${pids[@]}"
echo "Wait for all to complete..."
wait
echo "ALL DONE."
