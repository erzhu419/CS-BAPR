"""Native recovery training using the existing server-only SAC transport."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v144_multinode as transport


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(5, 6), required=True)
    parser.add_argument("--profile", choices=("no-pulse", "randomized-recovery"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    transport.run(parser.parse_args(), worker_filename="run_general_ood_v147_randomized_recovery.py")
