"""Native CUDA slow-actor probes; publish scalar JSON and server snapshots."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v132_multinode as v132

transport = v132.transport
PROFILES = ("friction-only", "gravity-only", "id")


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    command = [sys.executable, str(Path(__file__).resolve().parent/"run_general_ood_v141_actor_rate.py"),
               "--seed", str(args.seed), "--profile", args.profile, "--device", "cuda",
               "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        v132.publish(args)
    print(f"[PUBLISHED] seed{args.seed} {args.profile}: actor-rate scalars and server snapshots ready", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
