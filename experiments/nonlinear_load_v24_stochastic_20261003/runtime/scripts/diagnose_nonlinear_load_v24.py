"""Paired stochastic continuations with native normalized soft-return timing."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pysindy
import scipy
import sklearn
import torch
from torch.distributions import Normal

from scripts.diagnose_nonlinear_load_v23 import (
    ACTION_SCALE, BRANCHES, EP_LEN, GAMMA, NOISE_SEEDS, Q, R, RADII, SEVERITIES,
    SOURCES, STEPS, EnsembleQNet, EquilibriumGaussianPolicy, directional_scores,
    load_actor_parameters, make_radius_env, mean_direction,
)

REVISION = "nonlinear_load_v24_stochastic_diagnostic"
INPUT_REVISION = "nonlinear_load_v23_direction_diagnostic"
COSTS = ("total_cost", "discounted_cost", "state_cost", "command_cost", "early_state_cost",
         "late_state_cost", "discounted_state_cost", "discounted_command_cost")
RETURNS = ("normalized_reward_return", "future_entropy_bonus", "initial_entropy_bonus",
           "soft_q_return", "soft_actor_return")
SCORES = ("q_mean", "risk_penalty", "entropy_bonus", "risk_score", "unpenalized_soft_score", "soft_score")


def policy_sample(actor, states, noise, shift=0.):
    mean, log_std = actor(states)
    mean = mean + shift
    latent = mean + log_std.exp() * noise
    action0 = latent.tanh()
    log_prob = (Normal(mean, log_std.exp()).log_prob(latent)
                - torch.log(1 - action0.square() + 1e-6) - np.log(actor.action_range)).sum(-1)
    return actor.action_range * action0, log_prob


def stochastic_rollout(actors, critics, gains, terminals, radius, severity, seeds, noise_seeds):
    shape = (len(noise_seeds), len(seeds))
    raw = {mode: {"continuations": {branch: {k: np.zeros(shape) for k in COSTS + RETURNS} for branch in BRANCHES},
                  "critic_values": {branch: {k: np.zeros(shape) for k in SCORES} for branch in BRANCHES}}
           for mode in actors}
    initial, directions = None, {}
    for draw, noise_seed in enumerate(noise_seeds):
        generator = torch.Generator().manual_seed(noise_seed)
        environments = {(mode, branch): [make_radius_env(severity, seed, radius) for seed in seeds]
                        for mode in actors for branch in BRANCHES}
        observations = {key: np.stack([env.reset()[0] for env in envs]) for key, envs in environments.items()}
        if initial is None:
            initial = next(iter(observations.values())).copy()
            for mode, actor in actors.items():
                states = torch.as_tensor(initial, dtype=torch.float32, device=next(actor.parameters()).device)
                directions[mode] = mean_direction(actor, gains[mode], states)[2]
        for states in observations.values():
            np.testing.assert_array_equal(states, initial)
        for step in range(EP_LEN):
            noise = torch.randn((len(seeds), 2), generator=generator)
            applied = []
            for (mode, branch), envs in environments.items():
                actor = actors[mode]
                device = next(actor.parameters()).device
                states = torch.as_tensor(observations[mode, branch], dtype=torch.float32, device=device)
                shift = BRANCHES[branch] * directions[mode] if step == 0 else 0.
                with torch.no_grad():
                    action, log_prob = policy_sample(actor, states, noise.to(device), shift)
                    if step == 0:
                        q = critics[mode](states, action)
                        q_mean = q.mean(0).cpu().numpy().astype(float)
                        risk = (terminals[mode]["beta"] * q.std(0)).cpu().numpy().astype(float)
                        entropy = (-terminals[mode]["alpha"] * log_prob).cpu().numpy().astype(float)
                        values = {"q_mean": q_mean, "risk_penalty": risk, "entropy_bonus": entropy,
                                  "risk_score": q_mean + risk, "unpenalized_soft_score": q_mean + entropy,
                                  "soft_score": q_mean + risk + entropy}
                        for name, value in values.items():
                            raw[mode]["critic_values"][branch][name][draw] = value
                actions = action.cpu().numpy()
                physical = np.stack([env.state for env in envs])
                state_cost = np.einsum("bi,ij,bj->b", physical, Q, physical)
                command = ACTION_SCALE * actions.astype(float)
                command_cost = np.einsum("bi,ij,bj->b", command, R, command)
                transitions = [env.step(a) for env, a in zip(envs, actions)]
                assert all(t[2] == (step == EP_LEN - 1) and not t[3] for t in transitions)
                cost = -np.asarray([t[1] for t in transitions])
                np.testing.assert_allclose(cost, state_cost + command_cost, rtol=1e-12, atol=1e-12)
                totals = raw[mode]["continuations"][branch]
                for name, value in (("total_cost", cost), ("discounted_cost", GAMMA ** step * cost),
                                    ("state_cost", state_cost), ("command_cost", command_cost),
                                    ("early_state_cost" if step < 50 else "late_state_cost", state_cost),
                                    ("discounted_state_cost", GAMMA ** step * state_cost),
                                    ("discounted_command_cost", GAMMA ** step * command_cost)):
                    totals[name][draw] += value
                entropy = (-terminals[mode]["alpha"] * log_prob).cpu().numpy().astype(float)
                totals["initial_entropy_bonus" if step == 0 else "future_entropy_bonus"][draw] += GAMMA ** step * entropy
                observations[mode, branch] = np.stack([t[0] for t in transitions])
                applied.append(np.stack([t[4]["applied_disturbance"] for t in transitions]))
            for forces in applied[1:]:
                np.testing.assert_array_equal(applied[0], forces)
        for mode in actors:
            for totals in raw[mode]["continuations"].values():
                totals["normalized_reward_return"][draw] = -totals["discounted_cost"][draw] / terminals[mode]["reward_std"]
                totals["soft_q_return"][draw] = totals["normalized_reward_return"][draw] + totals["future_entropy_bonus"][draw]
                totals["soft_actor_return"][draw] = totals["soft_q_return"][draw] + totals["initial_entropy_bonus"][draw]
        print(f"radius={radius} load={severity} draw={draw + 1}/{len(noise_seeds)}: paired stochastic continuations; updates=0", flush=True)
    for mode in actors:
        for totals in raw[mode]["continuations"].values():
            assert all(np.isfinite(v).all() for v in totals.values())
            np.testing.assert_allclose(totals["total_cost"], totals["state_cost"] + totals["command_cost"])
            np.testing.assert_allclose(totals["discounted_cost"], totals["discounted_state_cost"] + totals["discounted_command_cost"])
    return initial, raw


def summarize_block(continuations, critic_values, analytic):
    derivatives, comparisons, analytic_comparisons = {}, {}, {}
    for h in STEPS:
        derivatives[str(h)] = {
            group: {key: (np.asarray(values[f"plus_{h}"][key]) - np.asarray(values[f"minus_{h}"][key])) / (2 * h)
                    for key in values["actor"]}
            for group, values in (("continuations", continuations), ("critic_values", critic_values))}
        actual = derivatives[str(h)]["continuations"]
        predicted = derivatives[str(h)]["critic_values"]
        comparisons[str(h)] = {}
        for score, target in (("q_mean", "normalized_reward_return"), ("q_mean", "soft_q_return"),
                              ("unpenalized_soft_score", "soft_actor_return"), ("soft_score", "soft_actor_return")):
            measured, prediction = actual[target], predicted[score]
            comparisons[str(h)][f"{score}_vs_{target}"] = {
                "improving_return_paths": int(np.sum(measured > 0)),
                "prediction_opposes_improving_paths": int(np.sum((measured > 0) & (prediction < 0))),
                "prediction_favors_worsening_paths": int(np.sum((measured < 0) & (prediction > 0))),
                "sign_disagreements": int(np.sum(measured * prediction < 0))}
        analytic_comparisons[str(h)] = {
            key: {"mean_abs_difference": float(np.abs(predicted[key] - analytic[key]).mean()),
                  "sign_disagreements": int(np.sum(predicted[key] * analytic[key] < 0))}
            for key in SCORES}
    return {"mean_directional_derivatives": {
                h: {group: {key: float(v.mean()) for key, v in values.items()} for group, values in groups.items()}
                for h, groups in derivatives.items()},
            "analytic_mean_score_derivatives": {k: float(np.asarray(v).mean()) for k, v in analytic.items()},
            "ordering": comparisons, "critic_fd_vs_analytic": analytic_comparisons,
            "finite_difference_step_sign_disagreements": {
                key: int(np.sum(derivatives[str(STEPS[0])]["continuations"][key]
                                * derivatives[str(STEPS[1])]["continuations"][key] < 0))
                for key in ("discounted_cost", "soft_q_return", "soft_actor_return")}}


def compact_model(raw, scores):
    means = {group: {branch: {k: v.mean(0).tolist() for k, v in values.items()}
                     for branch, values in branches.items()} for group, branches in raw.items()}
    analytic = {k: np.asarray(v).mean(0) for k, v in scores["expected"].items()}
    analytic["unpenalized_soft_score"] = analytic["q_mean"] + analytic["entropy_bonus"]
    analytic = {k: v.tolist() for k, v in analytic.items()}
    draws = {str(h): {group: {key: ((values[f"plus_{h}"][key] - values[f"minus_{h}"][key]) / (2 * h)).mean(1).tolist()
                             for key in values["actor"]} for group, values in raw.items()} for h in STEPS}
    return {**means, "analytic_score_derivatives": analytic,
            "latent_distance_to_fitted": scores["latent_distance_to_fitted"],
            "per_draw_mean_derivatives": draws,
            "summary": summarize_block(means["continuations"], means["critic_values"], analytic)}


def diagnose(payloads, source, device, count, radii, severities, noise_seeds, raw_dir):
    actors, critics, gains, terminals, before = {}, {}, {}, {}, {}
    for mode, payload in payloads.items():
        config = payload["source_protocol"]["config"]
        actors[mode] = EquilibriumGaussianPolicy(6, 2, config["hidden_dim"], activation="tanh").to(device).eval()
        critics[mode] = EnsembleQNet(6, 2, config["hidden_dim"], config["num_critics"]).to(device).eval()
        for net, key in ((actors[mode], "actor_parameters"), (critics[mode], "critic_parameters")):
            load_actor_parameters(net, payload[key])
            net.requires_grad_(False)
        gains[mode] = torch.as_tensor(payload["fitted_gain"], dtype=torch.float32, device=device)
        terminals[mode] = {k: payload[k] for k in ("alpha", "weighted_lambda", "reward_std")}
        terminals[mode]["beta"] = -2 - config["penalty_scale"] * payload["weighted_lambda"]
        before[mode] = [p.detach().clone() for net in (actors[mode], critics[mode]) for p in net.parameters()]
    torch.testing.assert_close(gains["floor"], gains["no-floor"], rtol=0, atol=0)
    raw_dir.mkdir(parents=True, exist_ok=True)
    seeds, blocks, raw_files = list(range(60000, 60000 + count)), {}, []
    for radius in radii:
        blocks[str(radius)] = {}
        for load in severities:
            initial, raw = stochastic_rollout(actors, critics, gains, terminals, radius, load, seeds, noise_seeds)
            previous = source["radius_results"][str(radius)][str(load)]
            np.testing.assert_array_equal(initial, previous["initial_observations"][:count])
            models = {}
            for mode in payloads:
                scores = directional_scores(actors[mode], critics[mode], gains[mode], initial,
                                            terminals[mode]["alpha"], terminals[mode]["beta"], noise_seeds)
                if count == 100:
                    for key in scores["expected"]:
                        np.testing.assert_allclose(scores["expected"][key], previous["models"][mode]["scores"]["expected"][key],
                                                   rtol=1e-4, atol=1e-5)
                models[mode] = compact_model(raw[mode], scores)
            raw_path = raw_dir / f"radius{radius}_load{load}.json"
            raw_path.write_text(json.dumps({"evaluation_seeds": seeds, "noise_seeds": list(noise_seeds),
                                           "models": {mode: {group: {branch: {k: v.tolist() for k, v in values.items()}
                                                                               for branch, values in branches.items()}
                                                              for group, branches in model.items()} for mode, model in raw.items()}},
                                          separators=(",", ":"), allow_nan=False) + "\n")
            raw_files.append({"path": str(raw_path.resolve()), "bytes": raw_path.stat().st_size})
            blocks[str(radius)][str(load)] = {"evaluation_seeds": seeds, "noise_seeds": list(noise_seeds),
                                              "initial_observations": initial.tolist(), "disturbance_paths_matched": True,
                                              "action_noise_matched": True, "models": models}
    for mode in payloads:
        params = [p for net in (actors[mode], critics[mode]) for p in net.parameters()]
        assert all(torch.equal(old, p) and p.grad is None for old, p in zip(before[mode], params))
    return {"radius_results": blocks, "terminal": terminals, "raw_files": raw_files,
            "parameters_unchanged": True, "optimizer_updates": 0, "fitting_interactions": 0,
            "controller_episodes": len(radii) * len(severities) * count * len(noise_seeds) * len(BRANCHES) * len(payloads)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    m = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert (m["revision"], m["input_revision"], m["radii"], m["severities"], m["finite_difference_steps"],
            m["noise_seeds"], m["evaluation_seed_start"], m["evaluation_episodes_per_block"], m["gamma"]) == (
        REVISION, INPUT_REVISION, list(RADII), list(SEVERITIES), list(STEPS), list(NOISE_SEEDS), 60000, 100, GAMMA)
    payloads = {mode: json.loads((args.input_dir / f"{mode}_seed18.json").read_text()) for mode in SOURCES}
    source = json.loads(args.source_result.read_text())
    assert source["revision"] == INPUT_REVISION and source["training_seed"] == 18 and not source["preflight_only"]
    assert source["configs"] == m["configs"] and source["environment_settings"] == m["environment_settings"]
    for mode, p in payloads.items():
        assert (p["revision"], p["source_mode"], p["source_id"], p["source_budget"]) == (
            INPUT_REVISION, mode, m["source_tasks"][mode]["id"], [500, 100000, 99001])
        assert p["source_protocol"]["config"] == m["configs"][mode]
        assert p["source_protocol"]["environment_settings"] == m["environment_settings"]
        assert all(p["source_protocol"]["config"][key] == 0 for key in ("weight_reg", "weight_sym", "policy_q_std_cap"))
        assert source["source_ids"][mode] == p["source_id"]
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for package, version in m["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert f"{sys.version_info.major}.{sys.version_info.minor}" in version
        else:
            assert versions[package] == version
    count, radii, loads, noise = (2, (1.,), (50,), NOISE_SEEDS[:2]) if args.preflight_only else (100, RADII, SEVERITIES, NOISE_SEEDS)
    out = {"revision": REVISION, "input_revision": INPUT_REVISION, "training_seed": 18,
           "source_ids": source["source_ids"], "configs": m["configs"], "environment_settings": m["environment_settings"],
           "finite_difference_steps": list(STEPS), "gamma": GAMMA, "checkpoint_selection": "last",
           "preflight_only": args.preflight_only, "runtime": {**versions, "python": sys.version, "device": args.device},
           **diagnose(payloads, source, args.device, count, radii, loads, noise, args.raw_dir)}
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed18.json"
    path.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Complete: {path}; raw draw/path evidence stays in {args.raw_dir}; updates=0", flush=True)


if __name__ == "__main__":
    main()
