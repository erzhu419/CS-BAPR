"""Stage only server-side final SAC ZIPs; publish small pulse evidence."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v130_multinode as transport

RUNTIME_REL = Path("experiments/general_ood_v146_pulse_recovery_20261014/runtime")
CASES = tuple(f"seed{s}_{p}" for s in (3, 4) for p in ("id", "gravity-1p5"))


def source_path(case):
    revision = "general_ood_v144_fixed_profile_sac_20261013" if case.endswith("_id") else "general_ood_v145_gravity_calibration_20261013"
    return transport.CENTRAL_PROJECT/f"experiments/{revision}/runtime/run_results/{case}/source.json"


def stage_source(case):
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(source_path(case)))], check=True, capture_output=True, text=True)
    row = json.loads(response.stdout)
    checkpoint = next(s["checkpoint"] for s in row["snapshots"] if s["step"] == 600000)
    row["evaluation_checkpoint"] = str(transport.copy_artifact(checkpoint))
    path = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/{case}/source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    return path


def publish(result_dir):
    path = result_dir/"pulse_recovery.json"
    destination = transport.CENTRAL_PROJECT/path.relative_to(transport.LOCAL_PROJECT)
    subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination.parent))], check=True)
    subprocess.run(["rsync", "-a", "-e", shlex.join(transport.ssh_args()), str(path), f"{transport.PEER}:{destination}"], check=True)


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = source_path(args.case) if central else stage_source(args.case)
    command = [sys.executable, str(Path(__file__).resolve().parent/"analyze_general_ood_v146_pulse_recovery.py"),
        "--case", args.case, "--source-json", str(source), "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args.result_dir)
    print(f"[PUBLISHED] {args.case}: pulse evidence ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
