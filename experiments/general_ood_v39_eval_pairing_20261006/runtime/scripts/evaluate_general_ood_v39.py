"""Re-evaluate v39 final policies with episode-seeded OOD randomness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import run_general_ood_v38 as v38
from scripts import run_general_ood_v39 as v39


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    for variant in v39.VARIANTS:
        checkpoint = args.checkpoint_dir / f"{v39.REVISION}_Hopper-v4_{variant}_seed1_final.pt"
        env = gym.make("Hopper-v4")
        agent = v39.make_agent("Hopper-v4", variant, "cuda", env)
        env.close()
        agent.load(str(checkpoint))
        count, updates = agent.context_tracker._count, agent.training_steps
        assert count == v39.BUDGET and updates == v39.BUDGET - agent.config.batch_size + 1
        modes, fixed = v38.evaluate(agent, "Hopper-v4", 20, 5)
        assert agent.context_tracker._count == count and agent.training_steps == updates
        ood = modes["markov_50x"]
        output = {
            "revision": v39.REVISION, "evaluation_revision": "episode-seeded-ood",
            "environment": "Hopper-v4", "variant": variant, "training_seed": 1,
            "checkpoint": checkpoint.name, "checkpoint_selection": "final",
            "training_budget": v39.BUDGET, "loaded_training_context_count": int(count),
            "loaded_training_update_steps": updates, "evaluation_only": True,
            "optimizer_updates_during_evaluation": 0,
            "restore_training_context_before_each_evaluation_block": True,
            "evaluation_seed_start": 80000, "evaluation_episodes": 20,
            "fixed_profile_episodes": 5, "ood_intensity": 50.,
            "ood_profile_family": "structured", "ood_schedule": "markov",
            "ood_switch_probability": 1., "ood_switch_steps": 200,
            "ood_rng_reseeded_on_episode_reset": True,
            "ood_non_id_transition_count": sum(n for p, n in ood["profile_counts"].items() if p != "id"),
            "ood_transition_count": sum(ood["lengths"]),
            "results": modes, "fixed_profile_results": fixed,
            "runtime": {"python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "gymnasium": gym.__version__,
                        "mujoco": __import__("mujoco").__version__, "gpu": torch.cuda.get_device_name()},
        }
        path = args.result_dir / f"{v39.REVISION}_Hopper-v4_{variant}_seed1_paired_eval.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"[DONE] paired evaluation {variant}: ID={modes['id']['mean']:.3f} "
              f"OOD={ood['mean']:.3f} non-ID-steps={output['ood_non_id_transition_count']} "
              f"result={path}", flush=True)
        del agent
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
