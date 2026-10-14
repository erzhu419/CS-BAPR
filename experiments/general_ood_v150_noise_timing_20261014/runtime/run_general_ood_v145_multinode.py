"""Native gravity calibration using the v144 server-only transport."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v144_multinode as transport


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=("gravity-1p5", "gravity-2p0"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    transport.run(parser.parse_args(), worker_filename="run_general_ood_v145_gravity_calibration.py")
