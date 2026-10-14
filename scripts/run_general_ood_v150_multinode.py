"""Transfer final700k ZIPs between servers; publish only noise-timing JSON."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v130_multinode as transport

RUNTIME_REL = Path("experiments/general_ood_v150_noise_timing_20261014/runtime")
CASES = tuple(f"seed{s}_{a}" for s in (5, 6) for a in ("no-pulse", "randomized-recovery"))


def source_path(case):
    return transport.CENTRAL_PROJECT/f"experiments/general_ood_v149_warm_recovery_20261014/runtime/run_results/{case}/source.json"


def stage_source(case):
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(source_path(case)))], check=True, capture_output=True, text=True)
    source = json.loads(response.stdout)
    snapshot = next(s for s in source["snapshots"] if s["step"] == 700000)
    source["evaluation_checkpoint"] = str(transport.copy_artifact(snapshot["checkpoint"]))
    destination = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/{case}/source.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(source, indent=2, allow_nan=False)+"\n")
    return destination


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = source_path(args.case) if central else stage_source(args.case)
    command = [sys.executable, str(Path(__file__).resolve().parent/"analyze_general_ood_v150_noise_timing.py"),
        "--case", args.case, "--source-json", str(source), "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        result = args.result_dir/"noise_timing.json"
        destination = transport.CENTRAL_PROJECT/result.relative_to(transport.LOCAL_PROJECT)
        subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination.parent))], check=True)
        subprocess.run(["rsync", "-a", "-e", shlex.join(transport.ssh_args()), str(result), f"{transport.PEER}:{destination}"], check=True)
    print(f"[PUBLISHED] {args.case}: frozen noise-timing evidence ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
