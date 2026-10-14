"""Stage final snapshots server-to-server and publish paired scalar results."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v134_multinode as v134

transport = v134.transport
RUNTIME_REL = Path("experiments/general_ood_v136_warmup_context_20261012/runtime")


def stage_source(seed):
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(v134.source_path(seed, "mse-control")))],
                              check=True, capture_output=True, text=True)
    row = json.loads(response.stdout)
    assert (row["revision"], row["training_seed"], row["arm"], row["smoke"]) == (
        "general_ood_v133_source_extension", seed, "mse-control", False)
    row["checkpoint"] = str(transport.copy_artifact(row["checkpoint"]))
    path = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/seed{seed}/source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    return path


def publish(args):
    path = args.result_dir/"warmup_context.json"
    row = json.loads(path.read_text())
    row["source"]["checkpoint"] = str(transport.CENTRAL_PROJECT/Path(row["source"]["checkpoint"]).relative_to(transport.LOCAL_PROJECT))
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    destination = transport.CENTRAL_PROJECT/path.relative_to(transport.LOCAL_PROJECT)
    subprocess.run([*transport.ssh_args(), transport.PEER, "mkdir -p "+shlex.quote(str(destination.parent))], check=True)
    subprocess.run(["rsync", "-a", "-e", shlex.join(transport.ssh_args()), str(path), f"{transport.PEER}:{destination}"], check=True)


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = v134.source_path(args.seed, "mse-control") if central else stage_source(args.seed)
    command = [sys.executable, str(Path(__file__).resolve().parent/"analyze_general_ood_v136_warmup_context.py"),
        "--seed", str(args.seed), "--source-json", str(source),
        "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args)
    print(f"[PUBLISHED] seed{args.seed}: paired warmup scalars ready", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
