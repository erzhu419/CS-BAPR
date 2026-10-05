"""Fresh-path confirmation of the unchanged v33 cost-linear reference."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy

from csbapr.envs.sustained_load_env import SustainedLoadEnv
from scripts.diagnose_nonlinear_load_v30 import control_actions, rollout
from scripts.diagnose_nonlinear_load_v32 import FittedInverseReference, local_actions, paired_costs
from scripts.diagnose_nonlinear_load_v33 import CostReference

REVISION = "nonlinear_load_v34_linear_confirmation"
STUDY = "nonlinear_load_v34_linear_confirmation_20261004"
TRAINING_SEEDS, RADII, SEVERITIES = (21, 22), (.2, .5, 1.), (1, 50)
CONTROLLERS = ("fitted_local", "cost_linear", "fitted_inverse", "known_inverse")
SEED_START, PREFLIGHT_START = 65000, 65500


def confirm(source, count, seed_start):
    fit = source["reference_identification"]["fits"]["cubic"]
    reference, inverse = CostReference(fit, source["environment_settings"]), FittedInverseReference(fit)
    np.testing.assert_allclose(reference.gain, source["source_candidate_gain"], rtol=1e-10, atol=1e-12)
    functions = {"fitted_local": lambda x: local_actions(inverse.gain, x),
                 "cost_linear": reference.linear_actions, "fitted_inverse": inverse,
                 "known_inverse": lambda x: control_actions("known_inverse", x)}
    seeds = list(range(seed_start, seed_start + count))
    results, reproduction = {}, {}
    for radius in RADII:
        results[str(radius)], reproduction[str(radius)] = {}, {}
        for severity in SEVERITIES:
            old = source["radius_results"][str(radius)][str(severity)]["controllers"]
            proof = {}
            for name in CONTROLLERS:
                previous = old[name]
                actual = rollout("sustained", name, radius, severity,
                                 previous["evaluation_seeds"], action_fn=functions[name])
                np.testing.assert_allclose(actual["returns"], previous["returns"], rtol=1e-5, atol=1e-6)
                proof[name] = {"evaluation_seeds": previous["evaluation_seeds"],
                               "maximum_return_error": float(np.max(np.abs(
                                   np.asarray(actual["returns"]) - previous["returns"])))}
            reproduction[str(radius)][str(severity)] = proof
            controls = {name: rollout("sustained", name, radius, severity, seeds, action_fn=fn)
                        for name, fn in functions.items()}
            for name in CONTROLLERS[1:]:
                for key in ("force_rms", "maximum_load_components", "force_change_counts"):
                    np.testing.assert_array_equal(controls[name][key], controls["fitted_local"][key])
            results[str(radius)][str(severity)] = {"controllers": controls,
                "linear_minus_baselines": {name: paired_costs(controls["cost_linear"], controls[name], seeds)
                                           for name in CONTROLLERS if name != "cost_linear"}}
            print(f"radius={radius} load={severity}: " + str({k:v["mean_cost"] for k,v in controls.items()}), flush=True)
    assert reference.action_components_solved == 0
    return {"reference_specification": {
                "gain": reference.gain.tolist(), "rule": "clip(-gain*observation, -1, 1)",
                "construction": "unchanged v33 CostReference.linear_actions",
                "load_persistence": reference.persistence, "command_cost": reference.command_cost.tolist(),
                "riccati_value": reference.value.tolist(), "origin_action_jacobian": (-reference.gain).tolist()},
            "source_candidate_gain_reproduced": True, "source_controls_reproduced": True,
            "references_changed": False, "reproduction": reproduction, "radius_results": results,
            "new_evaluation_episodes": 6 * count * len(CONTROLLERS), "reproduction_episodes": 48,
            "evaluation_interactions": (6 * count * len(CONTROLLERS) + 48) * 200,
            "nonlinear_action_components_solved": 0, "optimizer_updates": 0,
            "new_training_interactions": 0, "new_identification_interactions": 0, "checkpoint_access": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    source = json.loads(args.source_evaluation.read_text())
    assert (manifest["revision"], manifest["training_seeds"], manifest["controllers"], manifest["radii"],
            manifest["severities"], manifest["evaluation_seed_start"], manifest["preflight_seed_start"],
            manifest["evaluation_episodes_per_severity"]) == (
        REVISION, list(TRAINING_SEEDS), list(CONTROLLERS), list(RADII), list(SEVERITIES), SEED_START, PREFLIGHT_START, 100)
    assert (source["revision"], source["training_seed"], source["source_symbolic_constraint"]) == (
        manifest["source_revision"], args.seed, "on")
    assert source["source_config"] == manifest["source_config"]
    assert source["source_candidate_gain"] == manifest["source_candidate_gains"][str(args.seed)]
    assert source["environment_settings"] == manifest["environment_settings"] == SustainedLoadEnv.specification()
    assert np.__version__ == manifest["runtime_requirement"]["numpy"]
    assert scipy.__version__ == manifest["runtime_requirement"]["scipy"]
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    count, seed_start = (2, PREFLIGHT_START) if args.preflight_only else (100, SEED_START)
    started = time.monotonic()
    output = {"revision": REVISION, "source_revision": source["revision"], "training_seed": args.seed,
              "source_symbolic_constraint": "on", "source_config": source["source_config"],
              "environment_settings": source["environment_settings"], "preflight_only": args.preflight_only,
              "radii": list(RADII), "severities": list(SEVERITIES),
              "evaluation_seeds": list(range(seed_start, seed_start + count)),
              "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
              **confirm(source, count, seed_start)}
    output["elapsed_seconds"] = time.monotonic() - started
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}_diagnostic.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Linear reference confirmation complete: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
