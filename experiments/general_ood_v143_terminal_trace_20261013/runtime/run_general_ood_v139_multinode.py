"""Full replay moves server-to-server only; publish small diagnostic JSON."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v130_multinode as transport

RUNTIME_REL = Path("experiments/general_ood_v139_frozen_diagnostics_20261012/runtime")
CASES = ("v138_seed3_friction-only", "v138_seed4_gravity-only",
         "v138_seed4_friction-only", "v138_seed3_gravity-only", "v137_seed3_id", "v137_seed4_id")


def source_path(case):
    revision, seed, profile = case.split("_", 2)
    study = "physics_components" if revision == "v138" else "fixed_profile"
    return transport.CENTRAL_PROJECT/f"experiments/general_ood_{revision}_{study}_20261012/runtime/run_results/{seed}_{profile}/source.json"


def stage_source(case):
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(source_path(case)))],
                              check=True, capture_output=True, text=True)
    row = json.loads(response.stdout)
    row["resumable_checkpoint"] = str(transport.copy_artifact(row["resumable_checkpoint"]))
    path = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/{case}/source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    return path


def publish(result_dir):
    path = result_dir/"diagnostics.json"
    row = json.loads(path.read_text())
    local = Path(row["source"]["checkpoint"])
    row["source"]["checkpoint"] = str(transport.CENTRAL_PROJECT/local.relative_to(transport.LOCAL_PROJECT))
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    destination = transport.CENTRAL_PROJECT/path.relative_to(transport.LOCAL_PROJECT)
    subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination.parent))], check=True)
    subprocess.run(["rsync", "-a", "-e", shlex.join(transport.ssh_args()), str(path), f"{transport.PEER}:{destination}"], check=True)


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = source_path(args.case) if central else stage_source(args.case)
    command = [sys.executable, str(Path(__file__).resolve().parent/"analyze_general_ood_v139_frozen_diagnostics.py"),
               "--case", args.case, "--source-json", str(source), "--result-dir", str(args.result_dir)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args.result_dir)
    print(f"[PUBLISHED] {args.case}: diagnostic scalars only", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
