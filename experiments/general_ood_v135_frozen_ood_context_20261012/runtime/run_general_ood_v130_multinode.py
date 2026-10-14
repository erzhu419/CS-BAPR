"""Stage real v130 inputs between servers; run the unchanged learning/eval drivers."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

CENTRAL_PROJECT = Path("/home/zhengliang01/scheduleurm_work/CS-BAPR")
LOCAL_PROJECT = Path("/home/erzhu419/mine_code/CS-BAPR")
RUNTIME_REL = Path("experiments/general_ood_v130_independent_replication_20261011/runtime")
CENTRAL_RUNTIME = CENTRAL_PROJECT/RUNTIME_REL
PEER = "zhengliang01@202.197.46.16"


def ssh_args():
    return ["ssh", "-i", str(Path.home()/"scheduleurm_work/ssh/csbapr-peer"),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def native_environment():
    environment = os.environ.copy()
    libraries = Path(sys.prefix)/"lib/python3.10/site-packages/nvidia"
    paths = [str(p) for p in sorted(libraries.glob("*/lib"))]
    if environment.get("LD_LIBRARY_PATH"):
        paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(paths)
    return environment


def local_path(path):
    return LOCAL_PROJECT/Path(path).relative_to(CENTRAL_PROJECT)


def rewrite_paths(row, source, destination):
    for key in ("checkpoint", "source_checkpoint", "reference_checkpoint"):
        if key in row:
            row[key] = str(destination/Path(row[key]).relative_to(source))
    for entry in row.get("sources", []):
        entry["checkpoint"] = str(destination/Path(entry["checkpoint"]).relative_to(source))
    return row


def copy_artifact(path):
    destination = local_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--partial", "-e", shlex.join(ssh_args()),
                    f"{PEER}:{path}", str(destination)], check=True)
    return destination


def stage_source(path, full=False):
    completed = subprocess.run([*ssh_args(), PEER, "cat "+shlex.quote(str(path))],
                               check=True, capture_output=True, text=True)
    row = json.loads(completed.stdout)
    copy_artifact(row["checkpoint"])
    if full:
        copy_artifact(path.parents[2]/f"run_checkpoints/seed{row['training_seed']}/latest.pt")
    destination = local_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rewrite_paths(row, CENTRAL_PROJECT, LOCAL_PROJECT)
    with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as temporary:
        temporary.write(json.dumps(row, indent=2, allow_nan=False)+"\n")
    Path(temporary.name).replace(destination)


def prepare(seed, stage):
    source_runtime = CENTRAL_RUNTIME if seed >= 3 else CENTRAL_PROJECT/"experiments/general_ood_v123_source_extension_20261011/runtime"
    stage_source(source_runtime/f"run_results/seed{seed}/source.json", full=stage == "continuation")
    if stage == "evaluation":
        continuation_runtime = CENTRAL_RUNTIME if seed >= 3 else CENTRAL_PROJECT/"experiments/general_ood_v128_mean_retention_20261011/runtime"
        for arm in ("unchanged-control", "mean-only-source-regularization"):
            stage_source(continuation_runtime/f"run_results/seed{seed}_{arm}/mean_retention.json")


def publish(args):
    filename = "mean_retention.json" if args.stage == "continuation" else "replication.json"
    path = args.result_dir/filename
    row = json.loads(path.read_text())
    if args.stage == "continuation":
        final = Path(row["checkpoint"])
        remote_final = CENTRAL_PROJECT/final.relative_to(LOCAL_PROJECT)
        subprocess.run([*ssh_args(), PEER, "mkdir -p "+shlex.quote(str(remote_final.parent))], check=True)
        subprocess.run(["rsync", "-a", "--partial", "-e", shlex.join(ssh_args()), str(final),
                        f"{PEER}:{remote_final}"], check=True)
        row["resumable_checkpoint"] = str(args.checkpoint_dir/"latest.pt")
    rewrite_paths(row, LOCAL_PROJECT, CENTRAL_PROJECT)
    path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    remote_result = CENTRAL_PROJECT/path.relative_to(LOCAL_PROJECT)
    subprocess.run([*ssh_args(), PEER, "mkdir -p "+shlex.quote(str(remote_result.parent))], check=True)
    subprocess.run(["rsync", "-a", "--partial", "-e", shlex.join(ssh_args()), str(path),
                    f"{PEER}:{remote_result}"], check=True)


def run(args):
    central = Path(__file__).resolve().is_relative_to(CENTRAL_PROJECT)
    project = CENTRAL_PROJECT if central else LOCAL_PROJECT
    runtime = project/RUNTIME_REL
    if not central:
        print(f"[STAGE] seed{args.seed} {args.stage}: server-to-server inputs", flush=True)
        prepare(args.seed, args.stage)
    if args.stage == "continuation":
        script = "run_general_ood_v128_mean_retention.py"
        source = runtime if args.seed >= 3 else project/"experiments/general_ood_v123_source_extension_20261011/runtime"
        options = ["--arm", args.arm, "--source-runtime", str(source), "--device", "cuda"]
    else:
        script = "analyze_general_ood_v130_independent_replication.py"
        options = ["--v123-runtime", str(project/"experiments/general_ood_v123_source_extension_20261011/runtime"),
            "--v128-runtime", str(project/"experiments/general_ood_v128_mean_retention_20261011/runtime"),
            "--replication-runtime", str(runtime)]
    command = [sys.executable, str(runtime/script), "--seed", str(args.seed), *options,
               "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=None if central else native_environment())
    if not central:
        publish(args)
    print(f"[PUBLISHED] seed{args.seed} {args.stage}: final actor/scalars available in central store", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--stage", choices=("continuation", "evaluation"), required=True)
    parser.add_argument("--arm", choices=("unchanged-control", "mean-only-source-regularization"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "continuation":
        assert args.arm is not None
    run(args)


if __name__ == "__main__":
    main()
