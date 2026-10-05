"""Read-only correction of v35 evaluation with the actual reference mapping."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pysindy
import scipy
import sklearn
import torch

from scripts import test_lqr_convergence as training
from scripts.diagnose_nonlinear_load_v14 import paired_excess_costs, rollout
from scripts.diagnose_nonlinear_load_v9 import alignment
from scripts.train_nonlinear_load_v35 import (
    EVALUATION_SEED_START, MAPPINGS, METHOD, RADII, REFERENCES, SEVERITIES,
    ReferenceInterventionAgent, make_config, make_env,
)

REVISION = "nonlinear_load_v36_reference_alignment"
SOURCE_REVISION = "nonlinear_load_v35_reference_intervention"


def study_root():
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[1]):
        if (candidate / "dispatch_manifest.json").exists():
            return candidate
    raise FileNotFoundError("study dispatch_manifest.json not found")


def evaluate(agent, seeds, reference):
    agent.actor.eval()
    agent.actor.requires_grad_(False)
    gains = {
        "known_local": training.known_reference("nonlinear_load").to(agent.device).gain,
        "fitted": agent.pi_sym_torch.gain,
    }
    references = {"fitted": agent.pi_sym_torch}
    modes = ("actor", "fitted", "known_local", "known_inverse")
    results = {}
    for radius in RADII:
        results[str(radius)] = {}
        for severity in SEVERITIES:
            controllers, actor_samples = {}, None
            for mode in modes:
                result, samples = rollout(
                    agent.actor, gains, mode, radius, severity, seeds,
                    env_factory=make_env, references=references,
                )
                controllers[mode] = result
                if mode == "actor":
                    actor_samples = samples
            block = {
                "controllers": controllers,
                "paired_excess_costs": {
                    mode: paired_excess_costs(controllers["actor"], controllers[mode], seeds)
                    for mode in modes[1:]
                },
                "alignment": {
                    "initial_states": alignment(
                        agent.actor, gains, actor_samples[0], references=references
                    ),
                    "actor_trajectory": alignment(
                        agent.actor, gains, actor_samples.reshape(-1, 6), references=references
                    ),
                },
            }
            results[str(radius)][str(severity)] = block
    return {
        "radius_results": results,
        "feedback_gains": {
            **{name: gain.detach().cpu().tolist() for name, gain in gains.items()},
            "known_inverse_physical": training.get_optimal_gain().tolist(),
        },
        "evaluation_episodes_including_controls": len(RADII) * len(SEVERITIES) * len(seeds) * len(modes),
        "reference_mapping_verified": agent.pi_sym_torch.action_mapping == MAPPINGS[reference],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(21, 22))
    parser.add_argument("--reference", required=True, choices=REFERENCES)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--evaluation-seed-start", type=int, default=EVALUATION_SEED_START)
    parser.add_argument("--evaluation-count", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = study_root()
    manifest = json.loads((root / "dispatch_manifest.json").read_text())
    assert manifest["source_revision"] == SOURCE_REVISION
    assert manifest["source_task_ids"] == ["t134718", "t134719"]
    config = make_config("on", args.device)
    config.max_episodes = 500
    assert {**vars(config), "device": "cuda"} == manifest["config"]
    checkpoint = args.checkpoint_dir / f"{SOURCE_REVISION}_{args.reference}_seed{args.seed}_last.pt"
    assert checkpoint.exists(), f"missing completed checkpoint: {checkpoint}"
    agent = ReferenceInterventionAgent(config, args.seed, args.reference)
    agent.load(str(checkpoint))
    assert agent.reference_arm == args.reference
    assert agent.pi_sym_torch.action_mapping == MAPPINGS[args.reference]
    updates = agent.training_steps
    seeds = list(range(args.evaluation_seed_start, args.evaluation_seed_start + args.evaluation_count))
    evaluated = evaluate(agent, seeds, args.reference)
    assert agent.training_steps == updates
    output = {
        "revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "source_training_seed": args.seed,
        "source_reference": args.reference,
        "source_checkpoint": checkpoint.name,
        "reference_mapping": MAPPINGS[args.reference],
        "training_updates_at_evaluation": updates,
        "evaluation_seed_start": args.evaluation_seed_start,
        "evaluation_episodes_per_severity": args.evaluation_count,
        "evaluation_only": True,
        "optimizer_updates_during_evaluation": 0,
        "checkpoint_access": True,
        "reference_mapping_verified": True,
        **evaluated,
        "runtime": {
            "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "pysindy": pysindy.__version__, "sklearn": sklearn.__version__,
            "python": sys.version, "device": str(agent.device),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_{args.reference}_seed{args.seed}_evaluation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Corrected read-only evaluation saved to {path}", flush=True)


if __name__ == "__main__":
    main()
