#!/bin/bash
# Package CS-BAPR + MultiLineEnv for server deployment.
# Creates a tarball with all code, data, and instructions.

set -e
OUT=${OUT:-/tmp/csbapr_server.tar.gz}
STAGE=$(mktemp -d)

echo "Staging to $STAGE"

# ── CS-BAPR code ──
mkdir -p "$STAGE/CS-BAPR"
rsync -a --exclude='__pycache__' --exclude='*.pt' --exclude='*.pyc' \
      /home/erzhu419/mine_code/CS-BAPR/csbapr "$STAGE/CS-BAPR/"
rsync -a --exclude='__pycache__' \
      /home/erzhu419/mine_code/CS-BAPR/scripts "$STAGE/CS-BAPR/"

# ── offline-sumo env (MultiLineEnv + calibrated data) ──
mkdir -p "$STAGE/offline-sumo/env"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
      /home/erzhu419/mine_code/offline-sumo/env/sim_core "$STAGE/offline-sumo/env/"
rsync -a /home/erzhu419/mine_code/offline-sumo/env/calibrated_env \
      "$STAGE/offline-sumo/env/"

# ── requirements ──
cat > "$STAGE/requirements.txt" <<'EOF'
torch>=2.0
numpy>=1.24
pandas>=2.0
scipy>=1.10
openpyxl
efficient_kan
gymnasium
pygame
matplotlib>=3.5
EOF

# ── README ──
cat > "$STAGE/README.md" <<'EOF'
# CS-BAPR Server Experiments

## One-line deploy + run (recommended)

```bash
tar -xzf csbapr_server.tar.gz && cd csbapr_server && pip install -q -r requirements.txt && bash CS-BAPR/scripts/run_smart.sh
```

This auto-detects free GPU memory and CPU cores, schedules the 12 runs
adaptively, and produces `results.json`, `results_table.md`, and `figure1.pdf`
under `/tmp/csbapr_exp/` when finished.

## Setup (manual)
```bash
pip install -r requirements.txt
```

## Run options (manual)

**Smart (auto resource allocation)** — the recommended path, handles shared GPUs:
```bash
bash CS-BAPR/scripts/run_smart.sh
```

**Sequential (one run at a time, ~54h)**:
```bash
bash CS-BAPR/scripts/run_all_experiments.sh
```

**Parallel (all 12 at once, needs dedicated 12+ cores)**:
```bash
bash CS-BAPR/scripts/run_parallel.sh
```

## Custom env variables
- `LOGDIR` (default: /tmp/csbapr_exp) — where logs go
- `EPISODES` (default: 500) — training episodes per run
- `MEM_PER_RUN_MB` (default: 500) — GPU memory budget per run
- `CORES_PER_RUN` (default: 3) — CPU cores each run uses
- `RESERVE_CORE_FRAC` (default: 25) — % of cores kept free for other users

## Collect results
```bash
python scripts/collect_results.py --logdir /tmp/csbapr_exp --output results.json
```

Produces markdown tables:
- ID eval (total reward mean±std across seeds)
- OOD sweep (od_1x/2x/5x/10x/20x/50x)
- Abrupt shift (burst_5x/10x/20x/50x at t=3600s)

## Figure 1: Theoretical bound vs. actual (after runs complete)
```bash
python scripts/plot_figure1.py --logdir /tmp/csbapr_exp --output figure1.pdf
```

Overlays actual OOD reward degradation vs. CS-BAPR's Lean-proved bound
(Part X: δ + (ε + (L_eff+M)·d)·d) for paper's main theoretical result.

## Methods evaluated
- `csbapr` — NAU actor + training fixes (main method)
- `csbapr-kan` — KAN actor (ID-focused variant)
- `csbapr-no-nau` — MLP actor + fixes (ablation: isolates NAU contribution)
- `bapr` — plain MLP + vanilla SAC (baseline)

## Environment
MultiLineEnv loads 12 SUMO-calibrated bus lines from `offline-sumo/env/calibrated_env/`.
Training script: `CS-BAPR/scripts/test_multiline_convergence.py`

## Path setup
The scripts expect:
  CS-BAPR/            ← current directory
  offline-sumo/env/   ← parallel to CS-BAPR

Adjust `--env_path` if calibrated_env lives elsewhere.
EOF

# ── Tarball ──
cd "$(dirname "$STAGE")"
tar -czf "$OUT" -C "$STAGE" .
rm -rf "$STAGE"

echo ""
echo "Package ready: $OUT"
du -sh "$OUT"
tar -tzf "$OUT" | head -20
echo ""
echo "Total files:"
tar -tzf "$OUT" | wc -l
