"""Native SB3 execution; publish scalar evidence and server-side snapshots."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v130_multinode as transport

RUNTIME_REL = Path("experiments/general_ood_v144_fixed_profile_sac_20261013/runtime")


def publish(args):
    path = args.result_dir/"source.json"
    row = json.loads(path.read_text())
    snapshots = row["snapshots"]
    files = [Path(row["resumable_checkpoint"]), *(Path(r["checkpoint"]) for r in snapshots)]
    destination = transport.CENTRAL_PROJECT/files[0].parent.relative_to(transport.LOCAL_PROJECT)
    subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination))], check=True)
    subprocess.run(["rsync", "-a", "--partial", "-e", shlex.join(transport.ssh_args()),
        *(str(p) for p in files), f"{transport.PEER}:{destination}/"], check=True)
    row["resumable_checkpoint"] = str(destination/files[0].name)
    for snapshot in snapshots:
        snapshot["checkpoint"] = str(destination/Path(snapshot["checkpoint"]).name)
    row["training"]["snapshots"] = snapshots
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    remote = transport.CENTRAL_PROJECT/path.relative_to(transport.LOCAL_PROJECT)
    subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(remote.parent))], check=True)
    subprocess.run(["rsync", "-a", "-e", shlex.join(transport.ssh_args()), str(path), f"{transport.PEER}:{remote}"], check=True)


def run(args, worker_filename="run_general_ood_v144_fixed_profile_sac.py"):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    command = [sys.executable, str(Path(__file__).resolve().parent/worker_filename),
        "--seed", str(args.seed), "--profile", args.profile, "--device", "cuda",
        "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args)
    print(f"[PUBLISHED] seed{args.seed} {args.profile}: native SAC scalars ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=("friction-only", "gravity-only", "id"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
