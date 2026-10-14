"""Stage lightweight final policies server-to-server for read-only CPU traces."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v130_multinode as transport

RUNTIME_REL = Path("experiments/general_ood_v143_terminal_trace_20261013/runtime")
SOURCE_REL = Path("experiments/general_ood_v142_budget_extension_20261013/runtime")
PROFILES = ("friction-only", "gravity-only", "id")
CASES = [(s, p, p) for s in (3, 4) for p in PROFILES]+[(4, "id", p) for p in PROFILES[:2]]
CASE_NAMES = tuple(f"seed{s}_{source}_to_{target}" for s, source, target in CASES)


def source_path(case):
    seed, profile, _ = CASES[CASE_NAMES.index(case)]
    return transport.CENTRAL_PROJECT/SOURCE_REL/f"run_results/seed{seed}_{profile}/source.json"


def stage_source(case):
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(source_path(case)))],
                              check=True, capture_output=True, text=True)
    row = json.loads(response.stdout)
    assert row["revision"] == "general_ood_v142_budget_extension" and not row["smoke"]
    row["checkpoint"] = str(transport.copy_artifact(row["checkpoint"]))
    path = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/{case}/source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    return path


def publish(result_dir):
    path = result_dir/"terminal_trace.json"
    row = json.loads(path.read_text())
    row["source"]["checkpoint"] = str(transport.CENTRAL_PROJECT/Path(row["source"]["checkpoint"]).relative_to(transport.LOCAL_PROJECT))
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    destination = transport.CENTRAL_PROJECT/path.relative_to(transport.LOCAL_PROJECT)
    subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination.parent))], check=True)
    subprocess.run(["rsync", "-a", "-e", shlex.join(transport.ssh_args()), str(path), f"{transport.PEER}:{destination}"], check=True)


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = source_path(args.case) if central else stage_source(args.case)
    command = [sys.executable, str(Path(__file__).resolve().parent/"analyze_general_ood_v143_terminal_trace.py"),
        "--case", args.case, "--source-json", str(source), "--checkpoint-dir", str(args.checkpoint_dir),
        "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args.result_dir)
    print(f"[PUBLISHED] {args.case}: terminal-window scalar traces ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
