"""Co-locate a seed's three arms on one runtime, chosen by the scheduler."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_general_ood_v38 import BUDGETS, VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=BUDGETS)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    common = ["--env", args.env, "--seed", str(args.seed), "--stage", args.stage,
              "--checkpoint-dir", args.checkpoint_dir, "--result-dir", args.result_dir]
    worker = str(Path(__file__).with_name("run_general_ood_v38.py"))
    print(f"[BUNDLE] {args.env} seed{args.seed}: {','.join(VARIANTS)}", flush=True)
    processes = {variant: subprocess.Popen([sys.executable, "-u", worker,
                                            "--variant", variant, *common])
                 for variant in VARIANTS}
    codes = {variant: process.wait() for variant, process in processes.items()}
    print(f"[BUNDLE END] {args.env} seed{args.seed}: {codes}", flush=True)
    raise SystemExit(0 if all(code == 0 for code in codes.values()) else 1)


if __name__ == "__main__":
    main()
