"""Stage only final evaluation snapshots between Linux servers; publish scalars."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v133_multinode as v133

transport = v133.transport
RUNTIME_REL = Path("experiments/general_ood_v134_frozen_source_robustness_20261012/runtime")


def source_path(seed, arm):
    return transport.CENTRAL_PROJECT/v133.RUNTIME_REL/f"run_results/seed{seed}_{arm}/source.json"


def stage_source(seed, arm):
    path = source_path(seed, arm)
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(path))],
                              check=True, capture_output=True, text=True)
    row = json.loads(response.stdout)
    assert (row["revision"], row["training_seed"], row["arm"], row["smoke"]) == (
        "general_ood_v133_source_extension", seed, arm, False)
    row["checkpoint"] = str(transport.copy_artifact(row["checkpoint"]))
    destination = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/seed{seed}_{arm}/source.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    temporary.replace(destination)
    return destination


def publish(args):
    path = args.result_dir/"robustness.json"
    row = json.loads(path.read_text())
    row["source"]["checkpoint"] = str(transport.CENTRAL_PROJECT/Path(row["source"]["checkpoint"]).relative_to(transport.LOCAL_PROJECT))
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    destination = transport.CENTRAL_PROJECT/path.relative_to(transport.LOCAL_PROJECT)
    subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination.parent))], check=True)
    subprocess.run(["rsync", "-a", "-e", shlex.join(transport.ssh_args()),
                    str(path), f"{transport.PEER}:{destination}"], check=True)


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = source_path(args.seed, args.arm) if central else stage_source(args.seed, args.arm)
    command = [sys.executable, str(Path(__file__).resolve().parent/"analyze_general_ood_v134_frozen_robustness.py"),
        "--seed", str(args.seed), "--arm", args.arm, "--source-json", str(source),
        "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args)
    print(f"[PUBLISHED] seed{args.seed} {args.arm}: frozen CPU robustness scalars ready", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--arm", choices=v133.v132.ARMS, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
