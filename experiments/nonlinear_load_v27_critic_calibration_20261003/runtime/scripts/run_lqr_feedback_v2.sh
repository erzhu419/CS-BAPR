#!/usr/bin/env bash
set -euo pipefail

# Workers share the source directory, but never write the same run checkpoint.
# Serialize only the one-time dependency install on each worker host.
exec 9>.feedback_v2_runtime.lock
flock 9
if [[ ! -f .runtime/ready ]]; then
    base=""
    for candidate in \
            /home/erzhu419/anaconda3/envs/sumo-rl-mine/bin/python \
            /home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python \
            /home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python \
            "$HOME/scheduleurm_work/conda_envs/scomp-py310/bin/python" \
            "$HOME/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"; do
        if [[ -x "$candidate" ]] && "$candidate" -c \
                'import sys, torch; assert sys.version_info >= (3, 10); assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 4)' \
                >/dev/null 2>&1; then
            base="$candidate"
            break
        fi
    done
    if [[ -z "$base" ]]; then
        echo "No supported Python >=3.10 / PyTorch >=2.4 runtime on this worker" >&2
        exit 1
    fi
    if [[ ! -x .runtime/bin/python ]]; then
        "$base" -m venv .runtime
    fi
    # Reuse CUDA packages without exposing unrelated pandas/Arrow installations.
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
        'import torch, pysindy, numpy, scipy, sklearn, gym; assert torch.cuda.is_available(); assert torch.ones(1, device="cuda").cpu().numpy()[0] == 1'
    touch .runtime/ready
fi
flock -u 9
exec 9>&-
exec .runtime/bin/python -u scripts/test_lqr_convergence.py "$@"
