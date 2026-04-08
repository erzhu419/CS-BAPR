#!/bin/bash
# ============================================================
# CS-BAPR Full Experiment Suite
# ============================================================
#
# Runs all experiments needed for paper submission:
#   Phase 1: Training (all methods × all envs × 5 seeds)
#   Phase 2: OOD Evaluation (sweep + bound comparison)
#   Phase 3: Plot generation (Figures 1-4)
#
# Estimated time:
#   Pendulum: ~10 min/run × 7 methods × 5 seeds = ~6h
#   Hopper:   ~2h/run × 7 methods × 5 seeds = ~70h (use GPU)
#
# Usage:
#   # Quick test (Pendulum only, 1 seed, 50 episodes)
#   bash scripts/run_experiments.sh --quick
#
#   # Full Pendulum experiments
#   bash scripts/run_experiments.sh --env Pendulum-v1
#
#   # Full MuJoCo experiments (needs GPU)
#   bash scripts/run_experiments.sh --env Hopper-v4
#
#   # All environments
#   bash scripts/run_experiments.sh --full
#
#   # Only OOD evaluation (after training)
#   bash scripts/run_experiments.sh --eval-only
#
#   # Only plotting (after evaluation)
#   bash scripts/run_experiments.sh --plot-only

set -e

# ============ Configuration ============
SAVE_ROOT="${SAVE_ROOT:-results}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
EVAL_INTERVAL=50

# Methods: full CS-BAPR + ablations + baselines
METHODS="csbapr csbapr-relu csbapr-no-sindy csbapr-no-sym csbapr-no-jac bapr sac"

# Parse arguments
MODE="full"
ENVS="Pendulum-v1"
MAX_EPISODES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            SEEDS="0"
            MAX_EPISODES="--max-episodes 50"
            ENVS="Pendulum-v1"
            shift ;;
        --env)
            ENVS="$2"
            shift 2 ;;
        --full)
            ENVS="Pendulum-v1 Hopper-v4 HalfCheetah-v4"
            shift ;;
        --eval-only)
            MODE="eval"
            shift ;;
        --plot-only)
            MODE="plot"
            shift ;;
        --seeds)
            SEEDS="$2"
            shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

echo "=============================================="
echo "  CS-BAPR Experiment Suite"
echo "  Environments: $ENVS"
echo "  Methods: $METHODS"
echo "  Seeds: $SEEDS"
echo "  Save root: $SAVE_ROOT"
echo "  Mode: $MODE"
echo "=============================================="

cd "$(dirname "$0")/.."

# ============ Phase 1: Training ============
if [[ "$MODE" == "full" || "$MODE" == "train" ]]; then
    echo ""
    echo "===== PHASE 1: TRAINING ====="
    echo ""

    TOTAL=0
    DONE=0

    for env in $ENVS; do
        for method in $METHODS; do
            for seed in $SEEDS; do
                TOTAL=$((TOTAL + 1))
            done
        done
    done

    echo "Total runs: $TOTAL"
    echo ""

    for env in $ENVS; do
        echo "--- Environment: $env ---"
        for method in $METHODS; do
            PIDS=()
            for seed in $SEEDS; do
                run_name="${method}_${env}_${seed}"
                save_dir="${SAVE_ROOT}/${env}/${method}"
                log_file="${SAVE_ROOT}/logs/${run_name}.log"

                mkdir -p "${SAVE_ROOT}/logs" "$save_dir"

                # Skip if already done
                if [[ -f "${save_dir}/${run_name}_final.pt" ]]; then
                    echo "  [SKIP] $run_name (already exists)"
                    DONE=$((DONE + 1))
                    continue
                fi

                echo "  [RUN] $run_name"
                python scripts/train_csbapr.py \
                    --env "$env" \
                    --method "$method" \
                    --seed "$seed" \
                    --save-dir "$save_dir" \
                    --eval-interval $EVAL_INTERVAL \
                    $MAX_EPISODES \
                    > "$log_file" 2>&1 &

                PIDS+=($!)
                DONE=$((DONE + 1))

                # Limit parallelism
                if [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; then
                    wait "${PIDS[0]}"
                    PIDS=("${PIDS[@]:1}")
                fi
            done

            # Wait for remaining seeds of this method
            for pid in "${PIDS[@]}"; do
                wait "$pid"
            done
            echo "  [DONE] $method on $env (all seeds)"
        done
    done

    echo ""
    echo "Phase 1 complete: $DONE/$TOTAL runs"
fi

# ============ Phase 2: OOD Evaluation ============
if [[ "$MODE" == "full" || "$MODE" == "eval" ]]; then
    echo ""
    echo "===== PHASE 2: OOD EVALUATION ====="
    echo ""

    for env in $ENVS; do
        echo "--- OOD Sweep: $env ---"

        # Determine perturbation parameter and range
        case $env in
            Pendulum-v1)
                PARAM="mass"
                RANGE="0.5,1,2,5,10,20"
                ;;
            Hopper-v4)
                PARAM="body_mass"
                RANGE="0.5,1,2,4,8"
                ;;
            HalfCheetah-v4)
                PARAM="body_mass"
                RANGE="0.5,1,2,4,8"
                ;;
            Walker2d-v4)
                PARAM="body_mass"
                RANGE="0.5,1,2,4,8"
                ;;
            *)
                echo "  [SKIP] Unknown env: $env"
                continue
                ;;
        esac

        # Find best checkpoints
        NAU_CKPT=$(ls ${SAVE_ROOT}/${env}/csbapr/*_best.pt 2>/dev/null | head -1)
        RELU_CKPT=$(ls ${SAVE_ROOT}/${env}/bapr/*_best.pt 2>/dev/null | head -1)

        CKPT_ARGS=""
        if [[ -n "$NAU_CKPT" ]]; then
            CKPT_ARGS="$CKPT_ARGS --checkpoint-nau $NAU_CKPT"
        fi
        if [[ -n "$RELU_CKPT" ]]; then
            CKPT_ARGS="$CKPT_ARGS --checkpoint-relu $RELU_CKPT"
        fi

        # Protocol A: Static parameter sweep
        python scripts/ood_eval.py \
            --env "$env" \
            --mode sweep \
            --param "$PARAM" \
            --range "$RANGE" \
            --seeds 5 \
            --plot \
            $CKPT_ARGS \
            --output "${SAVE_ROOT}/${env}/ood_sweep_results.json" \
            2>&1 | tee "${SAVE_ROOT}/logs/ood_sweep_${env}.log"

        # Protocol B: Mid-episode abrupt shift (mirrors bus simulation)
        python scripts/ood_eval.py \
            --env "$env" \
            --mode shift \
            --param "$PARAM" \
            --range "$RANGE" \
            --shift-step 100 \
            --max-steps 500 \
            --seeds 5 \
            $CKPT_ARGS \
            --output "${SAVE_ROOT}/${env}/ood_shift_results.json" \
            2>&1 | tee "${SAVE_ROOT}/logs/ood_shift_${env}.log"

        echo "  [DONE] OOD evaluation for $env (sweep + shift)"
    done
fi

# ============ Phase 3: Plot Generation ============
if [[ "$MODE" == "full" || "$MODE" == "plot" ]]; then
    echo ""
    echo "===== PHASE 3: PLOT GENERATION ====="
    echo ""

    python scripts/plot_results.py --results-dir "$SAVE_ROOT" 2>&1 || echo "  [WARN] plot_results.py not yet implemented or failed"
fi

echo ""
echo "=============================================="
echo "  All phases complete!"
echo "  Results: $SAVE_ROOT/"
echo "=============================================="
