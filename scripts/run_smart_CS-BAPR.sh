#!/bin/bash
# Auto-resource-aware experiment runner.
# Detects free GPU memory + CPU cores, schedules 12 runs adaptively.
# Skips runs whose log already shows "Total time" (idempotent resumption).
# Interrupted runs automatically resume from the latest checkpoint
# (see CKPT_DIR below; test_multiline_convergence.py handles resume).

set -e
LOGDIR=${LOGDIR:-/tmp/csbapr_exp}
EPISODES=${EPISODES:-500}
MEM_PER_RUN_MB=${MEM_PER_RUN_MB:-500}   # Conservative GPU reservation per run
CORES_PER_RUN=${CORES_PER_RUN:-3}       # CPU cores each env simulation eats
RESERVE_CORE_FRAC=${RESERVE_CORE_FRAC:-25}  # Leave this % of cores for other users
# Checkpoint directory is inherited by each python invocation.
# Default lives beside the log directory so everything stays in one place.
export CKPT_DIR=${CKPT_DIR:-$LOGDIR/ckpt}
export CKPT_EVERY=${CKPT_EVERY:-10}

mkdir -p "$LOGDIR" "$CKPT_DIR"
cd "$(dirname "$0")/.."

# ── Detect resources ──
detect_gpu_slots() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo 12  # assume plenty if no CUDA
        return
    fi
    local total=0
    while read -r mem; do
        local s=$((mem / MEM_PER_RUN_MB))
        total=$((total + s))
    done < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    echo $total
}

pick_best_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo 0
        return
    fi
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | awk -F, '{gsub(/ /,""); print $1}'
}

total_cores=$(nproc)
usable_cores=$((total_cores * (100 - RESERVE_CORE_FRAC) / 100))
cpu_slots=$((usable_cores / CORES_PER_RUN))
gpu_slots=$(detect_gpu_slots)
max_parallel=$((gpu_slots < cpu_slots ? gpu_slots : cpu_slots))
[ $max_parallel -lt 1 ] && max_parallel=1

echo "===================================================="
echo "Resources: CPU cores=$total_cores (usable=$usable_cores)"
echo "           GPU slots=$gpu_slots (500MB each), CPU slots=$cpu_slots"
echo "           → max_parallel=$max_parallel"
echo "Logdir: $LOGDIR"
echo "Ckptdir: $CKPT_DIR (resume-aware, save every $CKPT_EVERY eps)"
echo "Episodes per run: $EPISODES"
echo "===================================================="

# ── Run queue ──
METHODS=("csbapr" "bapr" "csbapr-kan" "csbapr-no-nau")
SEEDS=(0 1 2)
pids=()

wait_for_slot() {
    while [ ${#pids[@]} -ge $max_parallel ]; do
        local new_pids=()
        for p in "${pids[@]}"; do
            kill -0 "$p" 2>/dev/null && new_pids+=("$p")
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -lt $max_parallel ] && break
        sleep 30
    done
}

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log="${LOGDIR}/${method}_seed${seed}.log"
        if [ -s "$log" ] && grep -q "Total time" "$log"; then
            echo "[SKIP ] $method seed=$seed (already complete)"
            continue
        fi
        wait_for_slot
        gpu=$(pick_best_gpu)
        # Append log marker so prior partial output (e.g. from a killed run)
        # is preserved; the python script will auto-resume from checkpoint.
        if [ -s "$log" ]; then
            echo "[LAUNCH] $method seed=$seed on GPU=$gpu → $log (appending, resume expected)"
            echo >> "$log"
            echo "===== RELAUNCH $(date -Iseconds) =====" >> "$log"
        else
            echo "[LAUNCH] $method seed=$seed on GPU=$gpu → $log (fresh)"
        fi
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$CORES_PER_RUN \
            python -u scripts/test_multiline_convergence.py \
                --episodes "$EPISODES" --seed "$seed" --method "$method" \
                >> "$log" 2>&1 &
        pids+=($!)
        sleep 10  # stagger: avoids GPU-memory spike during init overlap
    done
done

echo ""
echo "All jobs launched. Waiting for ${#pids[@]} background runs..."
wait

# ── Aggregate + plot ──
echo ""
echo "===== Collecting results ====="
python scripts/collect_results.py --logdir "$LOGDIR" \
    --output "${LOGDIR}/results.json" | tee "${LOGDIR}/results_table.md"

echo ""
echo "===== Generating Figure 1 ====="
python scripts/plot_figure1.py --logdir "$LOGDIR" \
    --output "${LOGDIR}/figure1.pdf" || echo "Figure 1 generation skipped (matplotlib?)"

echo ""
echo "===================================================="
echo "ALL DONE. Outputs in $LOGDIR:"
ls -la "$LOGDIR" | grep -E "\.(json|md|pdf|png|log)$"
echo "===================================================="
