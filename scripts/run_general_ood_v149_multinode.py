"""Reuse native source full states; publish warm-continuation evidence server-side."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v144_multinode as publication

transport = publication.transport
RUNTIME_REL = Path("experiments/general_ood_v149_warm_recovery_20261014/runtime")


def source_path(seed):
    return transport.CENTRAL_PROJECT/f"experiments/general_ood_v147_randomized_recovery_20261014/runtime/run_results/seed{seed}_no-pulse/source.json"


def stage_source(seed, profile, resume):
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(source_path(seed)))], check=True, capture_output=True, text=True)
    source = json.loads(response.stdout)
    native = transport.local_path(source["resumable_checkpoint"])
    if not resume and not native.exists():
        native = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/seed{seed}_{profile}/latest.pt"
        native.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--partial", "-e", shlex.join(transport.ssh_args()),
            f"{transport.PEER}:{source['resumable_checkpoint']}", str(native)], check=True)
    source["initialization_checkpoint"] = str(native)
    path = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/seed{seed}_{profile}/source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source, indent=2, allow_nan=False)+"\n")
    return path


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = source_path(args.seed) if central else stage_source(args.seed, args.profile, args.resume_from is not None)
    command = [sys.executable, str(Path(__file__).resolve().parent/"run_general_ood_v149_warm_recovery.py"),
        "--seed", str(args.seed), "--profile", args.profile, "--source-json", str(source), "--device", "cuda",
        "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publication.publish(args)
    print(f"[PUBLISHED] seed{args.seed} {args.profile}: warm continuation ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(5, 6), required=True)
    parser.add_argument("--profile", choices=("no-pulse", "randomized-recovery"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
