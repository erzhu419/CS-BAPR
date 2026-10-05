"""Frozen v25 critic directions against paired stochastic soft returns."""
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

from scripts.diagnose_nonlinear_load_v23 import (
    BRANCHES, GAMMA, NOISE_SEEDS, RADII, SEVERITIES, STEPS, EnsembleQNet,
    EquilibriumGaussianPolicy, directional_scores, load_actor_parameters, paired_rollout,
)
from scripts.diagnose_nonlinear_load_v24 import compact_model, stochastic_rollout

REVISION = "nonlinear_load_v26_critic_direction_diagnostic"
SOURCE_REVISION = "nonlinear_load_v25_critic_replay"
SOURCES, TRAINING_SEEDS = ("uniform", "early"), (19, 20)
BASELINE_KEYS = {"state_cost": "state_costs", "command_cost": "command_costs",
                 "early_state_cost": "early_state_costs", "late_state_cost": "late_state_costs"}


def export_checkpoint(path, evaluation_path, source_id, output):
    source = json.loads(evaluation_path.read_text())
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    extra, state = ckpt["extra_state"], ckpt["training_state"]
    protocol = extra["protocol"]
    assert protocol["revision"] == source["revision"] == SOURCE_REVISION
    assert protocol["training_seed"] == source["training_seed"] in TRAINING_SEEDS
    assert protocol["critic_sampling"] == source["critic_sampling"] in SOURCES
    assert protocol["phase"] == "arm" and not protocol["preflight_only"]
    assert all(source[k] == v for k, v in protocol.items())
    assert protocol["reference"] == "cubic-local" and protocol["jacobian_scope"] == "full"
    assert protocol["entropy_mode"] == "floor" and source["checkpoint_selection"] == "last"
    assert (extra["next_ep"], extra["env_steps"], ckpt["training_steps"]) == (500, 100000, 99001)
    assert (source["environment_steps"], source["training_updates"]) == (100000, 99001)
    gain = ckpt["symbolic_reference_gain"].tolist()
    assert gain == source["feedback_gains"]["fitted"]
    assert extra["reference_identification"] == source["reference_identification"]
    belief = state["belief"]
    weights = torch.tensor(belief["belief"], dtype=torch.float32)
    weighted_lambda = float((weights * torch.exp(-.1 * torch.arange(belief["max_H"], dtype=torch.float32))).sum())
    reward_std = max(float(state["reward_ema_var"]) ** .5, 1e-6)
    alpha = float(ckpt["alpha"])
    assert alpha == source["temperature_trace"][-1]["alpha"]
    assert reward_std == source["temperature_trace"][-1]["reward_std"]
    baselines = {}
    for radius in RADII:
        baselines[str(radius)] = {}
        for load in SEVERITIES:
            actor = source["radius_results"][str(radius)][str(load)]["controllers"]["actor"]
            assert actor["evaluation_seeds"] == list(range(60000, 60100))
            baselines[str(radius)][str(load)] = {
                "evaluation_seeds": actor["evaluation_seeds"],
                "costs": {"total_cost": (-np.asarray(actor["returns"])).tolist(),
                          **{k: actor[v] for k, v in BASELINE_KEYS.items()}}}
    payload = {"revision": REVISION, "source_id": source_id, "source_mode": protocol["critic_sampling"],
               "source_seed": protocol["training_seed"], "source_protocol": protocol,
               "source_budget": [500, 100000, 99001], "checkpoint_selection": "last",
               "actor_parameters": {k: v.tolist() for k, v in ckpt["actor"].items()},
               "critic_parameters": {k: v.tolist() for k, v in ckpt["critic"].items()},
               "fitted_gain": gain, "alpha": alpha, "weighted_lambda": weighted_lambda,
               "reward_std": reward_std, "source_baselines": baselines}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported {payload['source_mode']} seed{payload['source_seed']}: {output.stat().st_size} bytes; no replay/optimizer/checkpoint", flush=True)


def validate_inputs(payloads, manifest, seed):
    assert seed in TRAINING_SEEDS and set(payloads) == set(SOURCES)
    for mode, payload in payloads.items():
        assert (payload["revision"], payload["source_mode"], payload["source_seed"],
                payload["source_id"], payload["source_budget"], payload["checkpoint_selection"]) == (
                    REVISION, mode, seed, manifest["source_tasks"][str(seed)]["id"], [500, 100000, 99001], "last")
        protocol = payload["source_protocol"]
        assert protocol["revision"] == SOURCE_REVISION and protocol["training_seed"] == seed
        assert protocol["critic_sampling"] == mode and protocol["phase"] == "arm" and not protocol["preflight_only"]
        assert protocol["config"] == manifest["config"] and protocol["environment_settings"] == manifest["environment_settings"]
        assert (protocol["reference"], protocol["jacobian_scope"], protocol["entropy_mode"]) == ("cubic-local", "full", "floor")
        assert np.isfinite([payload[k] for k in ("alpha", "weighted_lambda", "reward_std")]).all()
        assert payload["alpha"] == .01 and 0 < payload["weighted_lambda"] <= 1 and payload["reward_std"] > 0
    assert payloads["uniform"]["fitted_gain"] == payloads["early"]["fitted_gain"]


def diagnose(payloads, device, count, radii, loads, noise_seeds, raw_dir):
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
    torch.testing.assert_close(gains["uniform"], gains["early"], rtol=0, atol=0)
    raw_dir.mkdir(parents=True, exist_ok=True)
    seeds, blocks, raw_files = list(range(60000, 60000 + count)), {}, []
    for radius in radii:
        blocks[str(radius)] = {}
        for load in loads:
            deterministic_initial, deterministic = paired_rollout(actors, gains, radius, load, seeds, steps=())
            reproduction = {}
            for mode, payload in payloads.items():
                baseline = payload["source_baselines"][str(radius)][str(load)]
                assert baseline["evaluation_seeds"][:count] == seeds
                gaps = {}
                for key, expected in baseline["costs"].items():
                    current = deterministic[mode]["actor"][key]
                    np.testing.assert_allclose(current, expected[:count], rtol=1e-4, atol=1e-3)
                    gaps[key] = float(np.max(np.abs(np.asarray(current) - expected[:count])))
                reproduction[mode] = {"passed": True, "maximum_absolute_cost_gaps": gaps}
            initial, raw = stochastic_rollout(actors, critics, gains, terminals, radius, load, seeds, noise_seeds)
            np.testing.assert_array_equal(initial, deterministic_initial)
            models = {}
            for mode in payloads:
                scores = directional_scores(actors[mode], critics[mode], gains[mode], initial,
                                            terminals[mode]["alpha"], terminals[mode]["beta"], noise_seeds)
                models[mode] = compact_model(raw[mode], scores)
            raw_path = raw_dir / f"radius{radius}_load{load}.json"
            raw_path.write_text(json.dumps({"evaluation_seeds": seeds, "noise_seeds": list(noise_seeds),
                "models": {mode: {group: {branch: {k: v.tolist() for k, v in values.items()}
                    for branch, values in branches.items()} for group, branches in model.items()}
                    for mode, model in raw.items()}}, separators=(",", ":"), allow_nan=False) + "\n")
            raw_files.append({"path": str(raw_path.resolve()), "bytes": raw_path.stat().st_size})
            blocks[str(radius)][str(load)] = {"evaluation_seeds": seeds, "noise_seeds": list(noise_seeds),
                "initial_observations": initial.tolist(), "disturbance_paths_matched": True,
                "action_noise_matched": True, "deterministic_reproduction": reproduction, "models": models}
    for mode in payloads:
        params = [p for net in (actors[mode], critics[mode]) for p in net.parameters()]
        assert all(torch.equal(old, p) and p.grad is None for old, p in zip(before[mode], params))
    return {"radius_results": blocks, "terminal": terminals, "raw_files": raw_files,
            "parameters_unchanged": True, "optimizer_updates": 0, "fitting_interactions": 0,
            "controller_episodes": len(radii) * len(loads) * count * len(noise_seeds) * len(BRANCHES) * len(payloads),
            "deterministic_reproduction_episodes": len(radii) * len(loads) * count * len(payloads)}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("checkpoint", "source-evaluation", "output"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--source-task", required=True)
    run = commands.add_parser("diagnose")
    run.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    for name in ("input-dir", "result-dir", "raw-dir"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        export_checkpoint(args.checkpoint, args.source_evaluation, args.source_task, args.output)
        return
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert (manifest["revision"], manifest["source_revision"], manifest["training_seeds"], manifest["modes"],
            manifest["radii"], manifest["severities"], manifest["finite_difference_steps"], manifest["noise_seeds"],
            manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_block"], manifest["gamma"]) == (
                REVISION, SOURCE_REVISION, list(TRAINING_SEEDS), list(SOURCES), list(RADII), list(SEVERITIES),
                list(STEPS), list(NOISE_SEEDS), 60000, 100, GAMMA)
    payloads = {mode: json.loads((args.input_dir / f"{mode}_seed{args.seed}.json").read_text()) for mode in SOURCES}
    validate_inputs(payloads, manifest, args.seed)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    assert all(versions[k] == v for k, v in manifest["runtime_requirement"].items() if k != "python_minor_versions")
    count, radii, loads, noise = (2, (1.,), (50,), NOISE_SEEDS[:2]) if args.preflight_only else (100, RADII, SEVERITIES, NOISE_SEEDS)
    out = {"revision": REVISION, "source_revision": SOURCE_REVISION, "training_seed": args.seed,
           "source_ids": {mode: p["source_id"] for mode, p in payloads.items()}, "config": manifest["config"],
           "environment_settings": manifest["environment_settings"], "finite_difference_steps": list(STEPS),
           "gamma": GAMMA, "checkpoint_selection": "last", "preflight_only": args.preflight_only,
           "runtime": {**versions, "python": sys.version, "device": args.device},
           **diagnose(payloads, args.device, count, radii, loads, noise, args.raw_dir)}
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}.json"
    path.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Complete: {path}; raw evidence stays in {args.raw_dir}; updates=0", flush=True)


if __name__ == "__main__":
    main()
