#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

exec 9>.v33_runtime.lock
flock 9
if [[ ! -f .runtime/ready ]]; then
    base=""
    for candidate in /usr/bin/python3 \
            "$HOME/.venvs/scheduleurm-torch-bench/bin/python" \
            /home/erzhu419/anaconda3/envs/assumption-agent/bin/python \
            /home/erzhu419/anaconda3/envs/sumo-rl-mine/bin/python \
            "$HOME"/scheduleurm_work/conda_envs/*/bin/python; do
        if [[ -x "$candidate" ]] && "$candidate" -c \
                'import sys, ensurepip; assert sys.version_info[:2] in ((3, 10), (3, 11))' >/dev/null 2>&1; then
            base="$candidate"
            break
        fi
    done
    [[ -n "$base" ]] || { echo "No Python 3.10/3.11 runtime on worker" >&2; exit 1; }
    "$base" -m venv .runtime
    .runtime/bin/python -m pip install --no-index \
        --find-links ../wheelhouse -r scripts/nonlinear_load_v30_requirements.txt
    .runtime/bin/python -c 'import numpy, scipy, gym, tqdm; assert numpy.__version__ == "2.2.6"; assert scipy.__version__ == "1.15.3"'
    touch .runtime/ready
fi
flock -u 9
exec 9>&-
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .runtime/bin/python -u scripts/diagnose_nonlinear_load_v33.py "$@"
