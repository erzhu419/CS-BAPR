#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

exec 9>.feedback_v2_runtime.lock
flock 9
if [[ ! -f .runtime/ready ]]; then
    base=""
    for candidate in \
            /home/erzhu419/anaconda3/envs/assumption-agent/bin/python \
            /home/erzhu419/anaconda3/envs/sumo-rl-mine/bin/python \
            /home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python \
            /home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python \
            "$HOME/scheduleurm_work/conda_envs/scomp-py310/bin/python" \
            "$HOME/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"; do
        if [[ -x "$candidate" ]] && "$candidate" -c \
                'import sys, torch; assert sys.version_info[:2] in ((3, 10), (3, 11)); assert torch.__version__ == "2.5.1+cu121"' \
                >/dev/null 2>&1; then
            base="$candidate"
            break
        fi
    done
    if [[ -z "$base" ]]; then
        echo "No Python 3.10/3.11 with Torch 2.5.1+cu121 runtime on this worker" >&2
        exit 1
    fi
    if [[ ! -x .runtime/bin/python ]]; then
        "$base" -m venv .runtime
    fi
    target_site=$(.runtime/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    "$base" - "$target_site" <<'PY'
import importlib.util
from pathlib import Path
import sys

target = Path(sys.argv[1])
for name in ("torch", "torchgen", "functorch", "nvidia", "triton", "filelock", "typing_extensions",
             "sympy", "mpmath", "networkx", "jinja2", "markupsafe", "fsspec"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        continue
    source = (Path(next(iter(spec.submodule_search_locations)))
              if spec.submodule_search_locations else Path(spec.origin))
    destination = target / (name if source.is_dir() else source.name)
    if not destination.exists():
        destination.symlink_to(source, target_is_directory=source.is_dir())
PY
    .runtime/bin/python -m pip install --no-index \
        --find-links ../wheelhouse -r scripts/lqr_feedback_v2_requirements.txt
    .runtime/bin/python -c \
        'import torch, pysindy, numpy, scipy, sklearn, gym; assert torch.__version__ == "2.5.1+cu121"; assert torch.cuda.is_available(); assert torch.ones(1, device="cuda").cpu().numpy()[0] == 1'
    touch .runtime/ready
fi
flock -u 9
exec 9>&-
exec .runtime/bin/python -u scripts/train_nonlinear_load_v29.py "$@"
