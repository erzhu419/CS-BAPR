#!/usr/bin/env bash
set -euo pipefail

base=""
for candidate in \
    "$HOME/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python" \
    /home/erzhu419/miniconda3/envs/csbapr/bin/python \
    /home/erzhu419/anaconda3/envs/assumption-agent/bin/python \
    /home/erzhu419/anaconda3/envs/sumo-rl-mine/bin/python \
    /home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python; do
    if [[ -x "$candidate" ]] && "$candidate" -c \
        'import sys, torch, gymnasium, mujoco; assert sys.version_info[:2] in ((3, 10), (3, 11)); assert torch.cuda.is_available()' \
        >/dev/null 2>&1; then
        base="$candidate"
        break
    fi
done

if [[ -z "$base" ]]; then
    echo "No compatible CUDA Python runtime found" >&2
    exit 1
fi

exec "$base" -u "$@"
