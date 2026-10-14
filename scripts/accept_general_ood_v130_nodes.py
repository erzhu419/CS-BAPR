"""Release existing queued v130 tasks after both native-node smokes have passed."""
import argparse
import json
import math
from pathlib import Path

NODES = ("jtl110gpu", "jtl110gpu2")
EXACT = ("model", "mode", "reset_seed", "noise_seed", "length", "physical_termination",
         "time_limit_truncation", "diagnostic_cutoff")
SCALARS = ("episode_return", "mean_velocity", "mean_policy_std", "mean_action_noise_l2", "mean_context_l2")


def validate(root):
    baseline = json.loads((root/"smoke_eval_seed2/replication.json").read_text())
    accepted = []
    for node in NODES:
        environment = json.loads((root/f"setup_{node}/environment.json").read_text())
        continuation = json.loads((root/f"smoke_{node}_continuation_seed2/mean_retention.json").read_text())
        evaluation = json.loads((root/f"smoke_{node}_evaluation_seed2/replication.json").read_text())
        assert environment["node"] == node and environment["environment_ready"]
        assert environment["runtime"] == baseline["runtime"] == evaluation["runtime"]
        assert continuation["training_seed"] == evaluation["training_seed"] == 2
        assert continuation["smoke"] and continuation["arm"] == "unchanged-control"
        assert continuation["actual_new_training_interactions"] == continuation["actual_new_optimizer_updates"] == 512
        assert continuation["training"]["interactions"] == 300512
        assert continuation["config"]["device"] == "cuda" and continuation["runtime"]["gpu"]
        assert continuation["mean_retention_coefficient"] == 0. and continuation["effective_beta"] == -2.
        assert evaluation["spec"] == baseline["spec"]
        assert evaluation["optimizer_updates"] == evaluation["new_training_interactions"] == 0
        assert len(evaluation["evaluations"]) == len(baseline["evaluations"]) == 6
        for actual, reference in zip(evaluation["evaluations"], baseline["evaluations"]):
            assert all(actual[k] == reference[k] for k in EXACT), (node, actual["model"], actual["mode"])
            for key in SCALARS:
                assert math.isclose(actual[key], reference[key], rel_tol=1e-7, abs_tol=1e-5), (node, key)
            assert actual["final_physical_state"]["unhealthy_conditions"] == reference["final_physical_state"]["unhealthy_conditions"]
        accepted.append(dict(node=node, runtime=environment["runtime"], gpu=continuation["runtime"]["gpu"],
            continuation_updates=512, frozen_evaluation_episodes=6, baseline_parity=True))
    return dict(status="passed", nodes=accepted, allowed_nodes=["node007", *NODES],
        preferred_node=None, preferred_gpu=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.result_dir)
    args.ready_file.write_text(json.dumps(result, indent=2)+"\n")
    print("[DONE] both Linux nodes passed native continuation and frozen-evaluation parity", flush=True)
