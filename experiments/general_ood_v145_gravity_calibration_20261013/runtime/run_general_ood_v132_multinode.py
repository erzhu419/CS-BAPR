"""Native v132 training; checkpoint publication is server-to-server only."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v130_multinode as transport

RUNTIME_REL = Path("experiments/general_ood_v132_source_stability_20261012/runtime")
ARMS = ("mse-control", "huber10")


def publish(args):
    path = args.result_dir/"source.json"
    row = json.loads(path.read_text())
    # Keep both the final evaluation actor and resumable replay on servers.
    for key in ("checkpoint", "resumable_checkpoint"):
        source = Path(row[key])
        destination = transport.CENTRAL_PROJECT/source.relative_to(transport.LOCAL_PROJECT)
        subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination.parent))], check=True)
        subprocess.run(["rsync", "-a", "--partial", "-e", shlex.join(transport.ssh_args()),
                        str(source), f"{transport.PEER}:{destination}"], check=True)
        row[key] = str(destination)
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    destination = transport.CENTRAL_PROJECT/path.relative_to(transport.LOCAL_PROJECT)
    subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination.parent))], check=True)
    subprocess.run(["rsync", "-a", "-e", shlex.join(transport.ssh_args()),
                    str(path), f"{transport.PEER}:{destination}"], check=True)


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    command = [sys.executable, str(Path(__file__).resolve().parent/"run_general_ood_v132_source_stability.py"),
        "--seed", str(args.seed), "--arm", args.arm, "--device", "cuda",
        "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args)
    print(f"[PUBLISHED] seed{args.seed} {args.arm}: source scalars and native snapshots ready", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
