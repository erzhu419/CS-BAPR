"""Frozen-actor ID-only Monte Carlo direction calibration of a critic clone."""
import argparse
import copy
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

from scripts.diagnose_nonlinear_load_v23 import (
    BRANCHES, EP_LEN, GAMMA, NOISE_SEEDS, RADII, SEVERITIES, STEPS, EnsembleQNet,
    EquilibriumGaussianPolicy, directional_scores, load_actor_parameters, mean_direction,
)
from scripts.diagnose_nonlinear_load_v24 import SCORES, policy_sample, stochastic_rollout, summarize_block
from scripts.diagnose_nonlinear_load_v26 import REVISION as SOURCE_REVISION

REVISION = "nonlinear_load_v27_critic_calibration"
TRAINING_SEEDS = (19, 20)
CALIBRATION_RADII = (.2, .5, 1.)
CALIBRATION_NOISE = tuple(range(93000, 93004))
CALIBRATION_SEED_START = 61000
H = .005
FIT_LR, ANCHOR_WEIGHT, GRAD_CLIP = .0003, 1., 1.


def calibration_spec(preflight):
    return {"radii": [1.] if preflight else list(CALIBRATION_RADII), "load": 1,
            "path_seeds": list(range(CALIBRATION_SEED_START, CALIBRATION_SEED_START + (2 if preflight else 16))),
            "noise_seeds": list(CALIBRATION_NOISE[:2] if preflight else CALIBRATION_NOISE),
            "h": H, "updates": 5 if preflight else 1000, "lr": FIT_LR,
            "anchor_weight": ANCHOR_WEIGHT, "grad_clip": GRAD_CLIP}


def validate_source(payload, source, manifest, seed):
    assert payload["revision"] == source["revision"] == manifest["source_revision"] == SOURCE_REVISION
    assert payload["source_seed"] == source["training_seed"] == seed in TRAINING_SEEDS
    assert payload["source_mode"] == "early" and payload["checkpoint_selection"] == source["checkpoint_selection"] == "last"
    assert payload["source_id"] == source["source_ids"]["early"] == manifest["source_tasks"][str(seed)]["training_id"]
    assert payload["source_budget"] == [500, 100000, 99001]
    protocol = payload["source_protocol"]
    assert protocol["revision"] == "nonlinear_load_v25_critic_replay" and protocol["training_seed"] == seed
    assert protocol["critic_sampling"] == "early" and protocol["phase"] == "arm" and not protocol["preflight_only"]
    assert protocol["config"] == source["config"] == manifest["config"]
    assert protocol["environment_settings"] == source["environment_settings"] == manifest["environment_settings"]
    assert (protocol["reference"], protocol["jacobian_scope"], protocol["entropy_mode"]) == ("cubic-local", "full", "floor")
    assert source["parameters_unchanged"] and source["optimizer_updates"] == source["fitting_interactions"] == 0
    assert not source["preflight_only"] and source["gamma"] == GAMMA and source["finite_difference_steps"] == list(STEPS)
    for key in ("alpha", "reward_std", "weighted_lambda"):
        assert source["terminal"]["early"][key] == payload[key]
    assert source["terminal"]["early"]["beta"] == -2 - manifest["config"]["penalty_scale"] * payload["weighted_lambda"]


def load_networks(payload, device):
    config = payload["source_protocol"]["config"]
    actor = EquilibriumGaussianPolicy(6, 2, config["hidden_dim"], activation="tanh").to(device).eval()
    critic = EnsembleQNet(6, 2, config["hidden_dim"], config["num_critics"]).to(device).eval()
    for net, key in ((actor, "actor_parameters"), (critic, "critic_parameters")):
        load_actor_parameters(net, payload[key])
        net.requires_grad_(False)
    gain = torch.as_tensor(payload["fitted_gain"], dtype=torch.float32, device=device)
    terminal = {k: payload[k] for k in ("alpha", "weighted_lambda", "reward_std")}
    terminal["beta"] = -2 - config["penalty_scale"] * terminal["weighted_lambda"]
    return actor, critic, gain, terminal


def collect_calibration(actor, critic, gain, terminal, spec, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    states, plus, minus, targets, target_draws, files = [], [], [], [], [], []
    for radius in spec["radii"]:
        initial, raw = stochastic_rollout({"early": actor}, {"early": critic}, {"early": gain}, {"early": terminal},
            radius, spec["load"], spec["path_seeds"], spec["noise_seeds"], {"plus": spec["h"], "minus": -spec["h"]})
        observations = torch.as_tensor(initial, dtype=torch.float32, device=next(actor.parameters()).device)
        direction = mean_direction(actor, gain, observations)[2]
        actions = {"plus": [], "minus": []}
        for noise_seed in spec["noise_seeds"]:
            noise = torch.randn((len(initial), 2), generator=torch.Generator().manual_seed(noise_seed)).to(observations.device)
            for branch, sign in (("plus", 1), ("minus", -1)):
                with torch.no_grad():
                    action, _ = policy_sample(actor, observations, noise, sign * spec["h"] * direction)
                    actions[branch].append(action.cpu())
        continuation = raw["early"]["continuations"]
        differences = (continuation["plus"]["soft_q_return"] - continuation["minus"]["soft_q_return"]) / (2 * spec["h"])
        states.append(observations.cpu())
        plus.append(torch.stack(actions["plus"]))
        minus.append(torch.stack(actions["minus"]))
        targets.append(torch.as_tensor(differences.mean(0), dtype=torch.float32))
        target_draws.append(differences)
        path = raw_dir / f"radius{radius}_load1.json"
        path.write_text(json.dumps({"radius": radius, "load": 1, "path_seeds": spec["path_seeds"],
            "noise_seeds": spec["noise_seeds"], "initial_observations": initial.tolist(),
            "continuations": {branch: {k: v.tolist() for k, v in values.items()} for branch, values in continuation.items()}},
            separators=(",", ":"), allow_nan=False) + "\n")
        files.append({"path": str(path.resolve()), "bytes": path.stat().st_size})
    draws = np.concatenate(target_draws, axis=1)
    metadata = {"spec": spec, "raw_files": files, "states": len(spec["radii"]) * len(spec["path_seeds"]),
                "controller_episodes": len(spec["radii"]) * len(spec["path_seeds"]) * len(spec["noise_seeds"]) * 2,
                "positive_direction_labels": int(np.sum(draws.mean(0) > 0)),
                "negative_direction_labels": int(np.sum(draws.mean(0) < 0)),
                "target_mean": float(draws.mean()), "mean_draw_standard_deviation": float(draws.std(0, ddof=1).mean()),
                "disturbance_paths_matched": True, "action_noise_matched": True}
    metadata["environment_interactions"] = metadata["controller_episodes"] * EP_LEN
    data = {"states": torch.cat(states), "plus_actions": torch.cat(plus, dim=1),
            "minus_actions": torch.cat(minus, dim=1), "targets": torch.cat(targets)}
    return data, metadata


def paired_predictions(critic, data):
    draws, count = data["plus_actions"].shape[:2]
    actions = torch.cat((data["plus_actions"].reshape(-1, 2), data["minus_actions"].reshape(-1, 2)))
    return critic(data["states"].repeat(2 * draws, 1), actions).reshape(critic.num_critics, 2, draws, count)


def fit_losses(prediction, original_prediction, targets, spec):
    means = prediction.mean(0)
    derivative = ((means[0] - means[1]) / (2 * spec["h"])).mean(0)
    direction = (derivative - targets).square().mean()
    anchor = (prediction - original_prediction).square().mean()
    return direction + spec["anchor_weight"] * anchor, direction, anchor


def save_fit(path, protocol, critic, optimizer, data, metadata, step, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"protocol": protocol, "critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
                "data": data, "calibration": metadata, "step": step, "history": history}, temporary)
    os.replace(temporary, path)


def fit_critic(original, data, metadata, protocol, checkpoint, resume=None, stop_after=None):
    calibrated = copy.deepcopy(original).requires_grad_(True).train()
    spec = protocol["calibration_spec"]
    optimizer = torch.optim.Adam(calibrated.parameters(), lr=spec["lr"])
    start, history = 0, []
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        assert state["protocol"] == protocol
        calibrated.load_state_dict(state["critic"])
        optimizer.load_state_dict(state["optimizer"])
        data, metadata, start, history = state["data"], state["calibration"], state["step"], state["history"]
    assert 0 <= start <= spec["updates"]
    device = next(original.parameters()).device
    tensors = {k: v.to(device) for k, v in data.items()}
    with torch.no_grad():
        anchor = paired_predictions(original, tensors)
    if not history:
        losses = fit_losses(paired_predictions(calibrated, tensors), anchor, tensors["targets"], spec)
        history.append({"step": 0, "total": float(losses[0].detach()), "direction": float(losses[1].detach()), "anchor": float(losses[2].detach())})
        save_fit(checkpoint, protocol, calibrated, optimizer, data, metadata, 0, history)
    end = spec["updates"] if stop_after is None else stop_after
    assert start <= end <= spec["updates"]
    for step in range(start + 1, end + 1):
        losses = fit_losses(paired_predictions(calibrated, tensors), anchor, tensors["targets"], spec)
        assert all(torch.isfinite(value) for value in losses)
        optimizer.zero_grad(set_to_none=True)
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(calibrated.parameters(), spec["grad_clip"])
        optimizer.step()
        if step % 100 == 0 or step == end:
            with torch.no_grad():
                values = fit_losses(paired_predictions(calibrated, tensors), anchor, tensors["targets"], spec)
            history.append({"step": step, "total": float(values[0]), "direction": float(values[1]), "anchor": float(values[2])})
            save_fit(checkpoint, protocol, calibrated, optimizer, data, metadata, step, history)
            print(f"fit {step}/{spec['updates']}: {history[-1]}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    calibrated.eval().requires_grad_(False)
    return calibrated, {"updates": end, "history": history,
        "parameter_change_l2": float(torch.sqrt(sum((a - b).square().sum() for a, b in zip(calibrated.parameters(), original.parameters()))))}, metadata


def rescore(actor, critics, gain, terminal, source):
    blocks = {}
    for radius in RADII:
        blocks[str(radius)] = {}
        for load in SEVERITIES:
            old_block = source["radius_results"][str(radius)][str(load)]
            assert old_block["evaluation_seeds"] == list(range(60000, 60100)) and old_block["noise_seeds"] == list(NOISE_SEEDS)
            measured = old_block["models"]["early"]
            states = torch.as_tensor(old_block["initial_observations"], dtype=torch.float32, device=next(actor.parameters()).device)
            direction = mean_direction(actor, gain, states)[2]
            models = {}
            for mode, critic in critics.items():
                raw = {branch: {key: [] for key in SCORES} for branch in BRANCHES}
                for noise_seed in NOISE_SEEDS:
                    noise = torch.randn((len(states), 2), generator=torch.Generator().manual_seed(noise_seed)).to(states.device)
                    for branch, shift in BRANCHES.items():
                        with torch.no_grad():
                            action, log_prob = policy_sample(actor, states, noise, shift * direction)
                            q = critic(states, action)
                            mean = q.mean(0).cpu().numpy().astype(float)
                            risk = (terminal["beta"] * q.std(0)).cpu().numpy().astype(float)
                            entropy = (-terminal["alpha"] * log_prob).cpu().numpy().astype(float)
                        values = {"q_mean": mean, "risk_penalty": risk, "entropy_bonus": entropy, "risk_score": mean + risk,
                                  "unpenalized_soft_score": mean + entropy, "soft_score": mean + risk + entropy}
                        for key, value in values.items():
                            raw[branch][key].append(value)
                raw = {b: {k: np.stack(v) for k, v in values.items()} for b, values in raw.items()}
                means = {b: {k: v.mean(0).tolist() for k, v in values.items()} for b, values in raw.items()}
                analytic_raw = directional_scores(actor, critic, gain, old_block["initial_observations"], terminal["alpha"], terminal["beta"], NOISE_SEEDS)
                analytic = {k: np.asarray(v).mean(0) for k, v in analytic_raw["expected"].items()}
                analytic["unpenalized_soft_score"] = analytic["q_mean"] + analytic["entropy_bonus"]
                analytic = {k: v.tolist() for k, v in analytic.items()}
                draws = {str(h): {k: ((raw[f"plus_{h}"][k] - raw[f"minus_{h}"][k]) / (2 * h)).mean(1).tolist()
                                  for k in SCORES} for h in STEPS}
                if mode == "original":
                    for branch in BRANCHES:
                        for key in SCORES:
                            np.testing.assert_allclose(means[branch][key], measured["critic_values"][branch][key], rtol=1e-4, atol=1e-5)
                models[mode] = {"critic_values": means, "analytic_score_derivatives": analytic,
                    "per_draw_score_mean_derivatives": draws,
                    "summary": summarize_block(measured["continuations"], means, analytic)}
            blocks[str(radius)][str(load)] = {"evaluation_seeds": old_block["evaluation_seeds"], "noise_seeds": list(NOISE_SEEDS),
                "initial_observations": old_block["initial_observations"], "latent_distance_to_fitted": measured["latent_distance_to_fitted"],
                "continuations": measured["continuations"], "per_draw_return_mean_derivatives": {
                    str(h): measured["per_draw_mean_derivatives"][str(h)]["continuations"] for h in STEPS}, "models": models}
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    for name in ("input-dir", "result-dir", "raw-dir", "ckpt-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert manifest["revision"] == REVISION and manifest["calibration"] == calibration_spec(False)
    assert manifest["preflight_calibration"] == calibration_spec(True) and manifest["training_seeds"] == list(TRAINING_SEEDS)
    payload = json.loads((args.input_dir / f"early_seed{args.seed}.json").read_text())
    source = json.loads((args.input_dir / f"v26_seed{args.seed}.json").read_text())
    validate_source(payload, source, manifest, args.seed)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    assert all(versions[k] == v for k, v in manifest["runtime_requirement"].items() if k != "python_minor_versions")
    actor, original, gain, terminal = load_networks(payload, args.device)
    before = [p.detach().clone() for net in (actor, original) for p in net.parameters()]
    protocol = {"revision": REVISION, "training_seed": args.seed, "source_training_id": payload["source_id"],
                "source_diagnostic_id": manifest["source_tasks"][str(args.seed)]["diagnostic_id"],
                "calibration_spec": calibration_spec(args.preflight_only), "config": manifest["config"],
                "environment_settings": manifest["environment_settings"], "preflight_only": args.preflight_only}
    checkpoint = args.ckpt_dir / f"{REVISION}_seed{args.seed}_last.pt"
    resume = args.resume or (checkpoint if checkpoint.exists() else None)
    if resume is None:
        data, metadata = collect_calibration(actor, original, gain, terminal, protocol["calibration_spec"], args.raw_dir)
    else:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        assert state["protocol"] == protocol
        data, metadata = state["data"], state["calibration"]
    calibrated, fit, metadata = fit_critic(original, data, metadata, protocol, checkpoint, resume)
    results = rescore(actor, {"original": original, "calibrated": calibrated}, gain, terminal, source)
    after = [p for net in (actor, original) for p in net.parameters()]
    assert all(torch.equal(a, b) and b.grad is None for a, b in zip(before, after))
    output = {**protocol, "source_revision": SOURCE_REVISION, "terminal": terminal, "calibration": metadata,
              "fit": fit, "radius_results": results, "actor_unchanged": True, "original_critic_unchanged": True,
              "optimizer_updates": fit["updates"], "online_training_interactions": 0,
              "new_evaluation_environment_interactions": 0, "measured_returns_reused_from_v26": True,
              "checkpoint_selection": "fixed-final-update", "checkpoint_path": str(checkpoint.resolve()),
              "runtime": {**versions, "python": sys.version, "device": args.device}}
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}.json"
    path.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Complete: {path}; actor unchanged; checkpoint/raw calibration data remain remote", flush=True)


if __name__ == "__main__":
    main()
