"""Finish the two native-version environments and stage the existing smoke inputs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import run_general_ood_v130_multinode as transfer

DATA_ROOTS = {"jtl110gpu": Path("/data/erzhu419/csbapr-v130"),
              "jtl110gpu2": Path("/mnt/data/erzhu419/csbapr-v130")}
WHEEL = "torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl"
WHEEL_URL = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl"


def run(args):
    started = time.monotonic()
    spec = json.loads((Path(__file__).resolve().parent/"node_setup_spec.json").read_text())
    base = DATA_ROOTS[args.node]
    environment = transfer.native_environment()
    environment["TMPDIR"] = str(base/"deployment/tmp")
    wheel = base/"deployment"/WHEEL
    if not zipfile.is_zipfile(wheel):
        print("[DOWNLOAD] official native Torch wheel", flush=True)
        subprocess.run(["curl", "-fL", "-C", "-", "--connect-timeout", "15",
            "--retry", "3", "--retry-delay", "3", "--max-time", "3600",
            "--output", str(wheel), WHEEL_URL], check=True, env=environment)
    pins = spec["cuda_pins"]
    dependencies = [f"{name}=={version}" for name, version in pins.items() if name != "torch"]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--timeout", "120",
        "--index-url", "https://pypi.org/simple", str(wheel), *dependencies], check=True, env=environment)
    code = ("import json,sys,torch,numpy,gymnasium,mujoco; "
            "print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,"
            "gymnasium=gymnasium.__version__,mujoco=mujoco.__version__)))")
    runtime = json.loads(subprocess.run([sys.executable, "-c", code], check=True,
        capture_output=True, text=True, env=environment).stdout)
    assert runtime == spec["runtime"], runtime
    installed = {name: importlib.metadata.version(name) for name in pins}
    assert installed == pins, installed
    transfer.prepare(2, "continuation")
    transfer.prepare(2, "evaluation")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    result = dict(node=args.node, runtime=runtime, cuda_pins=installed,
        environment_ready=True, gpu_smoke="pending", data_root=str(base),
        seconds=time.monotonic()-started)
    (args.result_dir/"environment.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"[DONE] {args.node} native-version environment and smoke inputs ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=DATA_ROOTS, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())
