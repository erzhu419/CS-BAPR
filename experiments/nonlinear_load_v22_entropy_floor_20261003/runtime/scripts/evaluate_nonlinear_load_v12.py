"""Evaluate frozen v11 inference parameters on new paths without loading replay state."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pysindy
import scipy
import sklearn
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
from csbapr.sindy.feedback import LQRFeedbackReference
from scripts.test_lqr_convergence import evaluate_feedback, evaluate_inverse_feedback, evaluate_ood, known_reference

REVISION = "nonlinear_load_v12_paths"
SEVERITIES = (1, 2, 5, 10, 20, 50)


def load_actor_parameters(actor, parameters):
    template = actor.state_dict()
    assert set(parameters) == set(template)
    actor.load_state_dict({name: torch.tensor(values, dtype=template[name].dtype,
                                             device=template[name].device)
                           for name, values in parameters.items()})
    actor.eval()


def evaluate_controls(agent, severities, count, seed_start):
    references = {"known_local_linear": known_reference("nonlinear_load").to(agent.device)}
    if agent.pi_sym_torch is not None:
        references["fitted_model"] = agent.pi_sym_torch
    controls = {name: evaluate_feedback(reference, severities, n_eval=count,
                                        seed_base=seed_start, environment="nonlinear_load")
                for name, reference in references.items()}
    controls["known_inverse"] = evaluate_inverse_feedback(severities, n_eval=count, seed_base=seed_start)
    return controls


def reproduce(agent, expected):
    actual = evaluate_ood(agent, [1, 50], n_eval=1, seed_base=50000, environment="nonlinear_load")
    controls = evaluate_controls(agent, [1, 50], 1, 50000)
    checks = {}
    for name, values in {"actor": actual, **{k: v["amplitude_results"] for k, v in controls.items()}}.items():
        checks[name] = {}
        for severity in (1, 50):
            value = values[severity]["returns"][0]
            previous = expected[name][str(severity)]
            np.testing.assert_allclose(value, previous, rtol=1e-4, atol=1e-3)
            checks[name][str(severity)] = {"actual": value, "source": previous, "return_gap": value - previous}
    return {"passed": True, "evaluation_seed": 50000, "severities": [1, 50],
            "rtol": 1e-4, "atol": 1e-3, "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-input", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    payload = json.loads(args.policy_input.read_text())
    protocol = payload["source_protocol"]
    assert protocol["revision"] == "nonlinear_load_v11_replication"
    assert protocol["training_seed"] in (7, 8) and protocol["environment"] == "nonlinear_load"
    assert (payload["next_episode"], payload["environment_steps"], payload["training_updates"]) == (500, 100000, 99001)
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    arm = next(a for a in manifest["arms"] if a["method"] == protocol["method"])
    assert protocol["config"] == arm["config"]
    assert protocol["environment_settings"] == manifest["environment_settings"]
    assert manifest["evaluation_seed_start"] == 60000 and manifest["evaluation_episodes_per_severity"] == 100
    assert manifest["severities"] == list(SEVERITIES)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    config = CSBAPRConfig(**{**protocol["config"], "device": args.device})
    assert not config.use_context and not config.use_nau_actor
    agent = CSBAPRAgent(config.state_dim, config.action_dim, config)
    assert str(agent.device) == args.device
    load_actor_parameters(agent.actor, payload["actor_parameters"])
    if arm["symbolic"]:
        assert payload["symbolic_reference_gain"] is not None
        agent.pi_sym_torch = LQRFeedbackReference(payload["symbolic_reference_gain"]).to(agent.device)
    else:
        assert payload["symbolic_reference_gain"] is None
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{REVISION}_{protocol['method']}_seed{protocol['training_seed']}"
    reproduction = reproduce(agent, payload["source_path_returns"])
    (args.result_dir / f"{tag}_reproduction.json").write_text(json.dumps(reproduction, indent=2, allow_nan=False) + "\n")
    print("Frozen-policy reproduction passed; optimizer updates=0", flush=True)
    if args.preflight_only:
        return
    results = evaluate_ood(agent, SEVERITIES, n_eval=100, seed_base=60000, environment="nonlinear_load")
    controls = evaluate_controls(agent, SEVERITIES, 100, 60000)
    assert agent.training_steps == 0 and agent.replay_buffer.size == 0
    assert all(math.isfinite(r) for v in results.values() for r in v["returns"])
    output = {**protocol, "revision": REVISION, "source_revision": protocol["revision"],
              "source_task_id": payload["source_task_id"], "source_node": payload["source_node"],
              "checkpoint_selection": "last", "environment_steps": payload["environment_steps"],
              "training_updates": payload["training_updates"], "actor_type": config.actor_type,
              "optimizer_steps_taken": 0, "new_training_interactions": 0, "new_identification_interactions": 0,
              "evaluation_seed_start": 60000, "evaluation_episodes_per_severity": 100,
              "sindy": payload["sindy_report"], "reproduction": reproduction,
              "runtime": {**versions, "python": sys.version, "device": str(agent.device),
                          "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None},
              "feedback_controls": controls, "amplitude_results": results}
    path = args.result_dir / f"{tag}_evaluation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Evaluation complete: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
