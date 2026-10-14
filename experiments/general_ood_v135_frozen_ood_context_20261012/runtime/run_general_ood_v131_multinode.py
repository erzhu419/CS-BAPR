"""Reuse v130's native nodes and direct server-to-server source transport."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v130_multinode as transport

RUNTIME_REL = Path("experiments/general_ood_v131_source_diagnostics_20261012/runtime")


def source_path(seed, steps):
    if seed < 3:
        runtime = transport.CENTRAL_PROJECT/"experiments/general_ood_v123_source_extension_20261011/runtime"
    else:
        runtime = transport.CENTRAL_RUNTIME
        if steps == 100000:
            runtime = runtime/"prefix"
    return runtime/f"run_results/seed{seed}/source.json"


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = source_path(args.seed, args.source_steps)
    project = transport.CENTRAL_PROJECT if central else transport.LOCAL_PROJECT
    if not central:
        print(f"[STAGE] seed{args.seed} age{args.source_steps}: server-to-server full state", flush=True)
        transport.stage_source(source, full=True)
        source = transport.local_path(source)
    command = [sys.executable, str(project/RUNTIME_REL/"probe_general_ood_v131_source_diagnostics.py"),
               "--seed", str(args.seed), "--source-steps", str(args.source_steps),
               "--source-json", str(source), "--result-dir", str(args.result_dir)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=None if central else transport.native_environment())
    if not central:
        path = args.result_dir/"source_diagnostics.json"
        row = json.loads(path.read_text())
        transport.rewrite_paths(row, transport.LOCAL_PROJECT, transport.CENTRAL_PROJECT)
        path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--source-steps", type=int, choices=(100000, 300000), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
