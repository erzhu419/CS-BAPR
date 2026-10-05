"""ID-cost-aware bounded control, evaluated without fitting or learner updates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
from scipy.linalg import solve_discrete_are

from csbapr.envs.sustained_load_env import SustainedLoadEnv
from scripts.diagnose_nonlinear_load_v30 import control_actions, rollout
from scripts.diagnose_nonlinear_load_v32 import FittedInverseReference, local_actions, paired_costs

REVISION = "nonlinear_load_v33_cost_reference"
STUDY = "nonlinear_load_v33_cost_reference_20261004"
TRAINING_SEEDS, RADII, SEVERITIES = (21, 22), (.2, .5, 1.), (1, 50)
CONTROLLERS = ("fitted_local", "fitted_inverse", "known_inverse", "cost_linear", "cost_nonlinear")
SEED_START = 64000
COMPARATORS = ("fitted_local", "cost_linear")


class CostReference:
    def __init__(self, fit, settings):
        self.dynamics = np.zeros((6, 6))
        self.dynamics[:4, :4] = fit["dynamics"]
        self.dynamics[:4, 4:] = fit["load_matrix"]
        self.persistence = 1 - settings["refresh_probability"]
        self.dynamics[4:, 4:] = self.persistence * np.eye(2)
        self.linear = np.vstack((fit["linear_action_matrix"], np.zeros((2, 2))))
        self.cubic = np.vstack((fit["cubic_action_matrix"], np.zeros((2, 2))))
        state_cost = np.diag(settings["state_cost_diagonal"] + [0., 0.])
        self.command_cost = (np.diag(settings["command_force_cost_diagonal"])
                             * settings["action_force_scale"] ** 2)
        self.value = solve_discrete_are(self.dynamics, self.linear, state_cost, self.command_cost)
        self.gain = np.linalg.solve(self.command_cost + self.linear.T @ self.value @ self.linear,
                                    self.linear.T @ self.value @ self.dynamics)
        products = (self.linear.T @ self.value @ self.linear,
                    self.linear.T @ self.value @ self.cubic,
                    self.cubic.T @ self.value @ self.cubic)
        for product in (*products, self.command_cost):
            np.testing.assert_allclose(product, np.diag(np.diag(product)), rtol=0, atol=1e-10)
        self.h11, self.h13, self.h33 = [np.diag(p) for p in products]
        self.r = np.diag(self.command_cost)
        assert np.all(self.h33 > 0) and np.all(self.r > 0)
        self.action_components_solved = 0
        self.maximum_surrogate_excess = 0.

    def linear_actions(self, states):
        return np.clip(-np.asarray(states) @ self.gain.T, -1., 1.)

    def terms(self, states):
        baseline = np.asarray(states) @ self.dynamics.T
        return baseline @ self.value @ self.linear, baseline @ self.value @ self.cubic

    def objective(self, states, actions):
        q1, q3 = self.terms(states)
        u = np.asarray(actions)
        return (2 * q1 * u + (self.r + self.h11) * u ** 2 + 2 * q3 * u ** 3
                + 2 * self.h13 * u ** 4 + self.h33 * u ** 6)

    def solve(self, states):
        q1, q3 = self.terms(states)
        actions = np.empty_like(q1)
        # Independent axes give quintic stationarity equations; compare every
        # real in-bound root and both endpoints, including nonconvex cases.
        for row in range(len(actions)):
            for axis in range(2):
                roots = np.polynomial.polynomial.polyroots([
                    q1[row, axis], self.r[axis] + self.h11[axis], 3 * q3[row, axis],
                    4 * self.h13[axis], 0., 3 * self.h33[axis]])
                interior = sorted(root.real for root in roots
                                  if abs(root.imag) < 1e-8 and -1 <= root.real <= 1)
                candidates = np.asarray([-1., 1., *interior])
                costs = (2 * q1[row, axis] * candidates
                         + (self.r[axis] + self.h11[axis]) * candidates ** 2
                         + 2 * q3[row, axis] * candidates ** 3
                         + 2 * self.h13[axis] * candidates ** 4
                         + self.h33[axis] * candidates ** 6)
                actions[row, axis] = candidates[np.argmin(costs)]
        return actions

    def __call__(self, states):
        actions = self.solve(states)
        excess = self.objective(states, actions) - self.objective(states, self.linear_actions(states))
        assert np.max(excess) <= 1e-9
        self.maximum_surrogate_excess = max(self.maximum_surrogate_excess, float(np.max(excess)))
        self.action_components_solved += actions.size
        return actions

    def jacobian(self, states):
        actions = self.solve(states)
        _, q3 = self.terms(states)
        curvature = (self.r + self.h11 + 6 * q3 * actions
                     + 12 * self.h13 * actions ** 2 + 15 * self.h33 * actions ** 4)
        interior = np.abs(actions) < 1 - 1e-10
        assert np.all(curvature[interior] > 1e-10)
        cross = self.linear.T[None] + 3 * actions[..., None] ** 2 * self.cubic.T[None]
        result = -(cross @ self.value @ self.dynamics) / curvature[..., None]
        return np.where(interior[..., None], result, 0.)

    def specification(self):
        return {"dynamics": self.dynamics.tolist(), "linear_action_matrix": self.linear.tolist(),
                "cubic_action_matrix": self.cubic.tolist(), "command_cost": self.command_cost.tolist(),
                "riccati_value": self.value.tolist(), "gain": self.gain.tolist(),
                "load_persistence": self.persistence, "origin_action_jacobian": (-self.gain).tolist(),
                "rule": "argmin_{u in [-1,1]^2} u^T R_command u + next(z,u)^T P next(z,u)",
                "model_source": "existing ID cubic physical fit and documented load conditional mean",
                "solver": "NumPy polynomial roots; all real stationary points and endpoints",
                "reference_is_optimal": False}


def diagnose(source, count):
    fit = source["reference_identification"]["fits"]["cubic"]
    reference, inverse = CostReference(fit, source["environment_settings"]), FittedInverseReference(fit)
    functions = {"fitted_local": lambda x: local_actions(inverse.gain, x), "fitted_inverse": inverse,
                 "known_inverse": lambda x: control_actions("known_inverse", x),
                 "cost_linear": reference.linear_actions, "cost_nonlinear": reference}
    np.testing.assert_allclose(reference.jacobian(np.zeros((1, 6)))[0], -reference.gain, atol=1e-12)
    seeds = list(range(SEED_START, SEED_START + count))
    results, reproduction = {}, {}
    for radius in RADII:
        results[str(radius)], reproduction[str(radius)] = {}, {}
        for severity in SEVERITIES:
            previous = source["radius_results"][str(radius)][str(severity)]["controllers"]
            proof = {}
            for name in CONTROLLERS[:3]:
                old = previous[name]
                actual = rollout("sustained", name, radius, severity, old["evaluation_seeds"], action_fn=functions[name])
                np.testing.assert_allclose(actual["returns"], old["returns"], rtol=1e-5, atol=1e-6)
                proof[name] = {"evaluation_seeds": old["evaluation_seeds"],
                               "maximum_return_error": float(np.max(np.abs(np.asarray(actual["returns"]) - old["returns"])))}
            reproduction[str(radius)][str(severity)] = proof
            controls = {name: rollout("sustained", name, radius, severity, seeds, action_fn=fn)
                        for name, fn in functions.items()}
            for name in CONTROLLERS[1:]:
                for key in ("force_rms", "maximum_load_components", "force_change_counts"):
                    np.testing.assert_array_equal(controls[name][key], controls["fitted_local"][key])
            results[str(radius)][str(severity)] = {"controllers": controls,
                "nonlinear_minus_baselines": {name: paired_costs(controls["cost_nonlinear"], controls[name], seeds)
                                              for name in COMPARATORS}}
            print(f"radius={radius} load={severity}: " + str({k:v["mean_cost"] for k,v in controls.items()}), flush=True)
    return {"reference_specification": reference.specification(), "matched_linear_origin_jacobian": True,
            "source_controls_reproduced": True, "reproduction": reproduction, "radius_results": results,
            "solver_accounting": {"action_components_solved": reference.action_components_solved,
                                  "maximum_surrogate_excess_vs_cost_linear": reference.maximum_surrogate_excess},
            "new_evaluation_episodes": 6 * count * len(CONTROLLERS), "reproduction_episodes": 36,
            "evaluation_interactions": (6 * count * len(CONTROLLERS) + 36) * 200,
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
    assert (manifest["revision"], manifest["training_seeds"], manifest["controllers"], manifest["radii"],
            manifest["severities"], manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"]) == (
        REVISION, list(TRAINING_SEEDS), list(CONTROLLERS), list(RADII), list(SEVERITIES), SEED_START, 100)
    assert (source["revision"], source["training_seed"], source["source_symbolic_constraint"]) == (
        manifest["source_revision"], args.seed, "on")
    assert source["source_config"] == manifest["source_config"]
    assert source["environment_settings"] == manifest["environment_settings"] == SustainedLoadEnv.specification()
    assert np.__version__ == manifest["runtime_requirement"]["numpy"]
    assert scipy.__version__ == manifest["runtime_requirement"]["scipy"]
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    started = time.monotonic()
    output = {"revision": REVISION, "source_revision": source["revision"], "training_seed": args.seed,
              "source_symbolic_constraint": "on", "source_config": source["source_config"],
              "environment_settings": source["environment_settings"], "preflight_only": args.preflight_only,
              "radii": list(RADII), "severities": list(SEVERITIES),
              "evaluation_seeds": list(range(SEED_START, SEED_START + (2 if args.preflight_only else 100))),
              "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
              **diagnose(source, 2 if args.preflight_only else 100)}
    output["elapsed_seconds"] = time.monotonic() - started
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}_diagnostic.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Cost reference diagnostic complete: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
