"""Read full v141 states on servers only, then publish continuation evidence."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v132_multinode as v132

transport = v132.transport
RUNTIME_REL = Path("experiments/general_ood_v142_budget_extension_20261013/runtime")
SOURCE_REL = Path("experiments/general_ood_v141_actor_rate_20261013/runtime")
PROFILES = ("friction-only", "gravity-only", "id")


def source_path(seed, profile):
    return transport.CENTRAL_PROJECT/SOURCE_REL/f"run_results/seed{seed}_{profile}/source.json"


def stage_source(seed, profile):
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(source_path(seed, profile)))],
                              check=True, capture_output=True, text=True)
    row = json.loads(response.stdout)
    assert (row["revision"], row["training_seed"], row["training_profile"], row["smoke"]) == (
        "general_ood_v141_actor_rate", seed, profile, False)
    row["resumable_checkpoint"] = str(transport.copy_artifact(row["resumable_checkpoint"]))
    path = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/seed{seed}_{profile}/source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    return path


def publish(args):
    path = args.result_dir/"source.json"
    row = json.loads(path.read_text())
    row["source_checkpoint"] = str(transport.CENTRAL_PROJECT/Path(row["source_checkpoint"]).relative_to(transport.LOCAL_PROJECT))
    row["source_json"] = str(source_path(args.seed, args.profile))
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    v132.publish(args)


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    source = source_path(args.seed, args.profile) if central else stage_source(args.seed, args.profile)
    command = [sys.executable, str(Path(__file__).resolve().parent/"run_general_ood_v142_budget_extension.py"),
        "--seed", str(args.seed), "--profile", args.profile, "--source-json", str(source), "--device", "cuda",
        "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args)
    print(f"[PUBLISHED] seed{args.seed} {args.profile}: budget continuation ready", flush=True)


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
