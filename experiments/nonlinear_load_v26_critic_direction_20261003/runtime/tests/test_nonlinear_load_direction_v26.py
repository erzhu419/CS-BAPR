import copy
import json

import numpy as np
import pytest
import torch

from scripts import diagnose_nonlinear_load_v26 as diagnostic
from scripts.summarize_nonlinear_load_v26 import validate


def payloads_for(seed):
    config = {"hidden_dim": 8, "num_critics": 3, "penalty_scale": 2.}
    manifest = {"revision": diagnostic.REVISION, "source_revision": diagnostic.SOURCE_REVISION,
                "source_tasks": {str(seed): {"id": f"task{seed}"}}, "config": config, "environment_settings": {}}
    payloads = {}
    for index, mode in enumerate(diagnostic.SOURCES):
        torch.manual_seed(index)
        actor = diagnostic.EquilibriumGaussianPolicy(6, 2, 8, activation="tanh")
        critic = diagnostic.EnsembleQNet(6, 2, 8, 3)
        protocol = {"revision": diagnostic.SOURCE_REVISION, "training_seed": seed, "critic_sampling": mode,
                    "phase": "arm", "preflight_only": False, "config": config, "environment_settings": {},
                    "reference": "cubic-local", "jacobian_scope": "full", "entropy_mode": "floor"}
        payloads[mode] = {"revision": diagnostic.REVISION, "source_mode": mode, "source_seed": seed,
            "source_id": f"task{seed}", "source_budget": [500, 100000, 99001], "checkpoint_selection": "last",
            "source_protocol": protocol, "actor_parameters": {k: v.tolist() for k, v in actor.state_dict().items()},
            "critic_parameters": {k: v.tolist() for k, v in critic.state_dict().items()},
            "fitted_gain": [[1., 0., .5, 0., .4, 0.], [0., 1., 0., .5, 0., .4]],
            "alpha": .01, "weighted_lambda": .2, "reward_std": 2. + index}
    return payloads, manifest


@pytest.mark.parametrize("seed", diagnostic.TRAINING_SEEDS)
def test_source_identity_rejects_wrong_seed_and_arm(seed):
    payloads, manifest = payloads_for(seed)
    diagnostic.validate_inputs(payloads, manifest, seed)
    for field, wrong in (("source_seed", 18), ("source_mode", "uniform"), ("source_id", "wrong-task")):
        changed = copy.deepcopy(payloads)
        changed["early"][field] = wrong
        with pytest.raises(AssertionError):
            diagnostic.validate_inputs(changed, manifest, seed)


def test_export_selects_last_arm_scalars_without_replay(tmp_path):
    payloads, _ = payloads_for(19)
    p = payloads["early"]
    protocol = p["source_protocol"]
    source = {**protocol, "checkpoint_selection": "last", "environment_steps": 100000, "training_updates": 99001,
              "feedback_gains": {"fitted": p["fitted_gain"]}, "reference_identification": {"reference": "fixture"},
              "temperature_trace": [{"alpha": .01, "reward_std": 3.}], "radius_results": {}}
    for radius in diagnostic.RADII:
        source["radius_results"][str(radius)] = {}
        for load in diagnostic.SEVERITIES:
            actor = {"evaluation_seeds": list(range(60000, 60100)), "returns": [-10.] * 100,
                     **{name: [2.] * 100 for name in diagnostic.BASELINE_KEYS.values()}}
            source["radius_results"][str(radius)][str(load)] = {"controllers": {"actor": actor}}
    checkpoint = {"extra_state": {"protocol": protocol, "next_ep": 500, "env_steps": 100000,
        "reference_identification": source["reference_identification"]}, "training_steps": 99001,
        "training_state": {"reward_ema_var": 9., "belief": {"max_H": 2, "belief": [.25, .75]},
                           "replay_buffer": "excluded", "optimizer": "excluded"},
        "symbolic_reference_gain": torch.tensor(p["fitted_gain"], dtype=torch.float64),
        "alpha": .01, "actor": {k: torch.tensor(v) for k, v in p["actor_parameters"].items()},
        "critic": {k: torch.tensor(v) for k, v in p["critic_parameters"].items()}}
    ckpt_path, evaluation, output = (tmp_path / name for name in ("last.pt", "evaluation.json", "input.json"))
    torch.save(checkpoint, ckpt_path)
    evaluation.write_text(json.dumps(source))
    diagnostic.export_checkpoint(ckpt_path, evaluation, "task19", output)
    result = json.loads(output.read_text())
    assert result["source_seed"] == 19 and result["source_mode"] == "early" and result["reward_std"] == 3.
    assert result["weighted_lambda"] == float((torch.tensor([.25, .75]) * torch.exp(-.1 * torch.arange(2))).sum())
    assert not {"training_state", "optimizer", "replay_buffer"}.intersection(result)
    source["critic_sampling"] = "uniform"
    evaluation.write_text(json.dumps(source))
    with pytest.raises(AssertionError):
        diagnostic.export_checkpoint(ckpt_path, evaluation, "task19", output)


def test_read_only_pair_reproduces_baselines_and_keeps_model_own_scales(tmp_path):
    torch.set_num_threads(1)
    payloads, manifest = payloads_for(20)
    actors, gains = {}, {}
    for mode, p in payloads.items():
        actors[mode] = diagnostic.EquilibriumGaussianPolicy(6, 2, 8, activation="tanh").eval()
        diagnostic.load_actor_parameters(actors[mode], p["actor_parameters"])
        gains[mode] = torch.tensor(p["fitted_gain"])
    _, reference = diagnostic.paired_rollout(actors, gains, 1., 50, [60000, 60001], steps=())
    for mode, p in payloads.items():
        p["source_baselines"] = {"1.0": {"50": {"evaluation_seeds": [60000, 60001],
            "costs": {k: reference[mode]["actor"][k] for k in ("total_cost", *diagnostic.BASELINE_KEYS)}}}}
    result = diagnostic.diagnose(payloads, "cpu", 2, (1.,), (50,), diagnostic.NOISE_SEEDS[:2], tmp_path)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": diagnostic.scipy.__version__,
                "pysindy": diagnostic.pysindy.__version__, "sklearn": diagnostic.sklearn.__version__}
    manifest["runtime_requirement"] = {**versions, "python_minor_versions": ['.'.join(diagnostic.sys.version.split()[0].split('.')[:2])]}
    data = {**result, "revision": diagnostic.REVISION, "source_revision": diagnostic.SOURCE_REVISION,
            "source_ids": {mode: p["source_id"] for mode, p in payloads.items()}, "training_seed": 20,
            "config": manifest["config"], "environment_settings": {}, "checkpoint_selection": "last",
            "preflight_only": True, "gamma": .99, "finite_difference_steps": list(diagnostic.STEPS),
            "runtime": {**versions, "python": diagnostic.sys.version, "device": "cuda"}}
    validate(json.loads(json.dumps(data)), manifest, payloads, preflight=True)
    assert data["controller_episodes"] == 40 and data["deterministic_reproduction_episodes"] == 4
    raw = json.loads((tmp_path / "radius1.0_load50.json").read_text())
    assert np.asarray(raw["models"]["early"]["continuations"]["actor"]["total_cost"]).shape == (2, 2)
    changed = copy.deepcopy(data)
    changed["terminal"]["early"]["reward_std"] = changed["terminal"]["uniform"]["reward_std"]
    with pytest.raises(AssertionError):
        validate(changed, manifest, payloads, preflight=True)
    payloads["early"]["source_baselines"]["1.0"]["50"]["costs"]["total_cost"][0] += 10
    with pytest.raises(AssertionError):
        diagnostic.diagnose(payloads, "cpu", 2, (1.,), (50,), diagnostic.NOISE_SEEDS[:2], tmp_path / "mismatch")
