"""Stage v132 full states server-to-server, run native v133, publish snapshots."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v132_multinode as v132

transport = v132.transport
RUNTIME_REL = Path("experiments/general_ood_v133_source_extension_20261012/runtime")


def source_path(seed, arm):
    return transport.CENTRAL_PROJECT/v132.RUNTIME_REL/f"run_results/seed{seed}_{arm}/source.json"


def stage_source(seed, arm):
    source = source_path(seed, arm)
    response = subprocess.run([*transport.ssh_args(), transport.PEER, "cat "+shlex.quote(str(source))],
                              check=True, capture_output=True, text=True)
    row = json.loads(response.stdout)
    assert (row["revision"], row["training_seed"], row["arm"], row["smoke"]) == (
        "general_ood_v132_source_stability", seed, arm, False)
    transport.copy_artifact(row["resumable_checkpoint"])
    row["resumable_checkpoint"] = str(transport.local_path(row["resumable_checkpoint"]))
    destination = transport.LOCAL_PROJECT/RUNTIME_REL/f"source_inputs/seed{seed}_{arm}/source.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    temporary.replace(destination)
    return destination


def publish(args):
    path = args.result_dir/"source.json"
    row = json.loads(path.read_text())
    row["source_checkpoint"] = str(transport.CENTRAL_PROJECT/Path(row["source_checkpoint"]).relative_to(transport.LOCAL_PROJECT))
    row["source_json"] = str(source_path(args.seed, args.arm))
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    v132.publish(args)


def run(args):
    central = Path(__file__).resolve().is_relative_to(transport.CENTRAL_PROJECT)
    if central:
        source = source_path(args.seed, args.arm)
    else:
        print(f"[STAGE] seed{args.seed} {args.arm}: full100k source server-to-server", flush=True)
        source = stage_source(args.seed, args.arm)
    command = [sys.executable, str(Path(__file__).resolve().parent/"run_general_ood_v133_source_extension.py"),
        "--seed", str(args.seed), "--arm", args.arm, "--source-json", str(source), "--device", "cuda",
        "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=transport.native_environment())
    if not central:
        publish(args)
    print(f"[PUBLISHED] seed{args.seed} {args.arm}: extended source and full state ready", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--arm", choices=v132.ARMS, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
