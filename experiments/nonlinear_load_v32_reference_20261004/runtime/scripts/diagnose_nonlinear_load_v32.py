"""Read-only actuator-aware reference diagnostic from the existing ID fit JSON."""
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

REVISION = "nonlinear_load_v32_reference"
STUDY = "nonlinear_load_v32_reference_20261004"
TRAINING_SEEDS = (21, 22)
RADII, SEVERITIES = (.2, .5, 1.), (1, 50)
CONTROLLERS = ("fitted_local", "fitted_inverse", "known_inverse")
SEED_START = 63000


class FittedInverseReference:
    def __init__(self, fit):
        self.gain = np.asarray(fit["gain"], dtype=float)
        load = np.asarray(fit["load_matrix"], dtype=float)
        matrices = [np.asarray(fit[k], dtype=float) for k in ("linear_action_matrix", "cubic_action_matrix")]
        coefficients = [np.linalg.lstsq(load, matrix, rcond=None)[0] for matrix in matrices]
        for matrix, coefficient in zip(matrices, coefficients):
            np.testing.assert_allclose(load @ coefficient, matrix, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(coefficient, np.diag(np.diag(coefficient)), rtol=0, atol=1e-5)
        self.linear, self.cubic = [np.diag(c) for c in coefficients]
        assert self.gain.shape == (2, 6) and np.all(self.linear > 0) and np.all(self.cubic > 0)
        self.cap = self.linear + self.cubic
        self.force_gain = self.linear[:, None] * self.gain

    def inverse(self, force):
        # Closed-form real inverse of a*u + c*u^3, for positive ID-fitted a,c.
        scale = np.sqrt(self.linear / (3 * self.cubic))
        return 2 * scale * np.sinh(np.arcsinh(3 * force / (2 * self.linear * scale)) / 3)

    def __call__(self, states):
        desired = self.cap * np.tanh(-states @ self.force_gain.T / self.cap)
        return self.inverse(desired)

    def jacobian(self, states):
        bounded = np.tanh(-states @ self.force_gain.T / self.cap)
        action = self.inverse(self.cap * bounded)
        force_jacobian = -(1 - bounded ** 2)[..., None] * self.force_gain
        slope = self.linear + 3 * self.cubic * action ** 2
        return force_jacobian / slope[..., None]

    def specification(self):
        return {"linear_coefficients": self.linear.tolist(), "cubic_coefficients": self.cubic.tolist(),
                "maximum_force_from_fit": self.cap.tolist(), "force_gain": self.force_gain.tolist(),
                "gain": self.gain.tolist(), "origin_action_jacobian": (-self.gain).tolist(),
                "rule": "inverse_ID_cubic(cap*tanh(-diag(linear)*gain*observation/cap))",
                "fit_source": "existing ID-only cubic fit; no oracle plant matrices or new fit"}


def local_actions(gain, states):
    return np.tanh(-states @ np.asarray(gain, dtype=np.float32).T)


def paired_costs(candidate, baseline, seeds):
    differences = -np.asarray(candidate["returns"]) + np.asarray(baseline["returns"])
    return {"mean_total_cost_change": float(differences.mean()),
            "better_paths": int(np.sum(differences < 0)), "per_path_cost_changes": differences.tolist(),
            "evaluation_seeds": seeds,
            "mean_state_cost_change": float(np.mean(np.sum(np.asarray(candidate["state_costs_by_window"])
                                                              - baseline["state_costs_by_window"], axis=1))),
            "mean_command_cost_change": float(np.mean(np.sum(np.asarray(candidate["command_costs_by_window"])
                                                                - baseline["command_costs_by_window"], axis=1)))}


def initial_jacobians(reference, radius, severity, seeds):
    envs = [SustainedLoadEnv(severity, seed) for seed in seeds]
    for env in envs:
        env.amplitude = radius
    states = np.stack([env.reset()[0] for env in envs])
    local = local_actions(reference.gain, states)
    local_jacobian = -(1 - local ** 2)[..., None] * reference.gain
    inverse = reference(states)
    inverse_jacobian = reference.jacobian(states)
    metrics = {}
    for name, action, jacobian in (("fitted_local", local, local_jacobian),
                                   ("fitted_inverse", inverse, inverse_jacobian)):
        force_jacobian = (reference.linear + 3 * reference.cubic * action ** 2)[..., None] * jacobian
        metrics[name] = {"mean_output_force_load_jacobian": force_jacobian[:, :, 4:].mean(0).tolist(),
                         "load_cancellation_jacobian_residual_sq": float(np.mean(np.sum(
                             (np.eye(2) + force_jacobian[:, :, 4:]) ** 2, axis=(1, 2))))}
    return {"states": len(states), "controls": metrics,
            "action_jacobian_difference_sq": float(np.mean(np.sum((inverse_jacobian - local_jacobian) ** 2, (1, 2))))}


def diagnose(source, count):
    reference = FittedInverseReference(source["reference_identification"]["fits"]["cubic"])
    seeds = list(range(SEED_START, SEED_START + count))
    reproduction, results = {}, {}
    functions = {"fitted_local": lambda x: local_actions(reference.gain, x),
                 "fitted_inverse": reference, "known_inverse": lambda x: control_actions("known_inverse", x)}
    origin = reference.jacobian(np.zeros((1, 6)))[0]
    np.testing.assert_allclose(origin, -reference.gain, rtol=1e-12, atol=1e-12)
    for radius in RADII:
        reproduction[str(radius)], results[str(radius)] = {}, {}
        for severity in SEVERITIES:
            old = source["radius_results"][str(radius)][str(severity)]["controllers"]
            proof = {}
            for name, original in (("fitted_local", "fitted"), ("known_inverse", "known_inverse")):
                old_seeds = old[original]["evaluation_seeds"][:2]
                actual = rollout("sustained", name, radius, severity, old_seeds, action_fn=functions[name])
                expected = old[original]["returns"][:2]
                np.testing.assert_allclose(actual["returns"], expected, rtol=1e-5, atol=1e-6)
                proof[name] = {"evaluation_seeds": old_seeds,
                               "maximum_return_error": float(np.max(np.abs(np.asarray(actual["returns"]) - expected)))}
            reproduction[str(radius)][str(severity)] = proof
            controls = {name: rollout("sustained", name, radius, severity, seeds, action_fn=functions[name])
                        for name in CONTROLLERS}
            for name in CONTROLLERS[1:]:
                for key in ("force_rms", "maximum_load_components", "force_change_counts"):
                    np.testing.assert_array_equal(controls[name][key], controls["fitted_local"][key])
            results[str(radius)][str(severity)] = {
                "controllers": controls,
                "inverse_minus_local": paired_costs(controls["fitted_inverse"], controls["fitted_local"], seeds),
                "initial_jacobians": initial_jacobians(reference, radius, severity, seeds)}
            print(f"radius={radius} load={severity}: " + str({m: c["mean_cost"] for m, c in controls.items()}), flush=True)
    return {"reference_specification": reference.specification(), "origin_jacobian_preserved": True,
            "source_controls_reproduced": True, "reproduction": reproduction, "radius_results": results,
            "new_evaluation_episodes": 6 * count * 3, "reproduction_episodes": 24,
            "evaluation_interactions": (6 * count * 3 + 24) * 200,
            "optimizer_updates": 0, "new_training_interactions": 0, "new_identification_interactions": 0,
            "checkpoint_access": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    source = json.loads(args.source_evaluation.read_text())
    assert (source["revision"], source["training_seed"], source["symbolic_constraint"],
            source["training_updates"], source["environment_steps"], source["checkpoint_selection"]) == (
        manifest["source_revision"], args.seed, "on", 99001, 100000, "last")
    assert (manifest["revision"], manifest["training_seeds"], manifest["radii"], manifest["severities"],
            manifest["controllers"], manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"]) == (
        REVISION, list(TRAINING_SEEDS), list(RADII), list(SEVERITIES), list(CONTROLLERS), SEED_START, 100)
    assert source["environment_settings"] == manifest["environment_settings"] == SustainedLoadEnv.specification()
    assert source["config"] == manifest["source_config"]
    assert np.__version__ == manifest["runtime_requirement"]["numpy"]
    assert scipy.__version__ == manifest["runtime_requirement"]["scipy"]
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    count, started = (2 if args.preflight_only else 100), time.monotonic()
    output = {"revision": REVISION, "source_revision": source["revision"], "training_seed": args.seed,
              "source_symbolic_constraint": "on", "environment_settings": source["environment_settings"],
              "source_config": source["config"], "radii": list(RADII), "severities": list(SEVERITIES),
              "evaluation_seeds": list(range(SEED_START, SEED_START + count)), "preflight_only": args.preflight_only,
              "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
              **diagnose(source, count)}
    output["elapsed_seconds"] = time.monotonic() - started
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}_diagnostic.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Reference diagnostic complete: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
