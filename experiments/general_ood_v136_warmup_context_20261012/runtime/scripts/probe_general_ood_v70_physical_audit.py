"""Privileged physical-template diagnostic on fresh, frozen-policy OOD paths."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v69_counterfactual as v69
from scripts.v68_mode_likelihood import MODES, observe_series

REVISION = "general_ood_v70_physical_audit"
SPEC = dict(seeds=[135070, 135071], modes=["normal", "high_grav", "compound"], steps=6000, switch_step=3800)
CHECKPOINT_STEPS = 1000
ORACLE_KEYS = ("oracle_observations", "oracle_physical_states", "actual_physical_states",
               "integration_states", "elapsed_steps", "clipped_input_velocities")


def audit_cells(spec):
    return [dict(seed=seed, mode=mode) for seed in spec["seeds"] for mode in spec["modes"]]


def active_mode(cell, step, spec):
    return "normal" if step < spec["switch_step"] else cell["mode"]


def apply_mode(env, mode):
    v69.apply_mode_to_env(env, "Hopper-v4", v69.MUJOCO_MODE_PROFILES["Hopper-v4"][mode])


def physical_state(env):
    return np.concatenate((env.unwrapped.data.qpos, env.unwrapped.data.qvel)).copy()


def trace_path(directory, cell):
    return directory/f"seed{cell['seed']}_{cell['mode']}.npz"


def collect(agent, env, shadows, cell, spec, loop=None, checkpoint=None):
    if loop is None:
        random.seed(cell["seed"])
        np.random.seed(cell["seed"])
        torch.manual_seed(cell["seed"])
        apply_mode(env, "normal")
        state, _ = env.reset(seed=cell["seed"])
        agent.reset_context()
        loop = dict(step=0, state=state, episode=0, initial_updates=agent.training_steps,
            initial_replay_size=agent.replay_buffer.size,
            rows={k: [] for k in (*v69.v68.v53.TRACE_KEYS, *ORACLE_KEYS)})
    for step in range(loop["step"], spec["steps"]):
        mode = active_mode(cell, step, spec)
        if step == spec["switch_step"]:
            apply_mode(env, mode)
        state = loop["state"]
        action = agent.select_action(state, deterministic=False)
        executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
        snapshot = v69.environment_state(env)
        clipped = bool(np.any(np.abs(env.unwrapped.data.qvel) > 10.))
        actual = env.step(executed)
        actual_physical = physical_state(env)
        observations, physical = [], []
        for candidate, shadow in shadows.items():
            v69.restore_step_state(shadow, snapshot)
            np.testing.assert_array_equal(shadow.unwrapped._get_obs(), state)
            values = shadow.step(executed)
            next_physical = physical_state(shadow)
            if candidate == mode:
                np.testing.assert_array_equal(values[0], actual[0])
                np.testing.assert_array_equal(next_physical, actual_physical)
                assert values[1:4] == actual[1:4], "true-mode-shadow-not-exact"
            observations.append(values[0].copy())
            physical.append(next_physical)
        v69.append_transition(loop["rows"], state, action, executed, actual)
        for key, value in zip(ORACLE_KEYS, (observations, physical, actual_physical,
                snapshot["values"], snapshot["elapsed_steps"], clipped)):
            loop["rows"][key].append(value)
        next_state, reward, terminated, truncated, _ = actual
        done = terminated or truncated
        agent.observe_context(state, action, reward, next_state, done)
        loop["step"], loop["state"] = step+1, next_state
        if done and step+1 < spec["steps"]:
            loop["episode"] += 1
            loop["state"], _ = env.reset(seed=cell["seed"]+10000*loop["episode"])
            agent.reset_context()
        if checkpoint is not None and (step+1) % CHECKPOINT_STEPS == 0:
            checkpoint(loop)
            print(f"[AUDIT] seed{cell['seed']} {cell['mode']} {step+1}/{spec['steps']}", flush=True)
    assert agent.training_steps == loop["initial_updates"], "audit-updated-policy"
    assert agent.replay_buffer.size == loop["initial_replay_size"], "oracle-entered-RL-replay"
    profile = v69.MUJOCO_MODE_PROFILES["Hopper-v4"][cell["mode"]]
    raw = env.unwrapped.model
    np.testing.assert_array_equal(raw.body_mass, env._original_body_mass*profile["body_mass"])
    np.testing.assert_array_equal(raw.geom_friction, env._original_friction*profile["friction"])
    np.testing.assert_array_equal(raw.opt.gravity, env._original_gravity*profile["gravity"])
    trace = {k: np.asarray(v) for k, v in loop["rows"].items()}
    record = dict(**cell, actual_transitions=spec["steps"], counterfactual_step_calls=5*spec["steps"],
        environment_step_calls=6*spec["steps"], exact_true_mode_replays=spec["steps"], optimizer_updates=0,
        physical_terminations=int(trace["terminated"].sum()), time_limit_truncations=int(trace["truncated"].sum()),
        clipped_input_velocities=int(trace["clipped_input_velocities"].sum()))
    return trace, record


def distribution(values):
    return dict(mean=float(np.mean(values)), minimum=float(np.min(values)),
        p10=float(np.quantile(values, .1)), median=float(np.median(values)), p90=float(np.quantile(values, .9)))


def summarize_trace(trace, cell, reference, spec):
    bank = reference.bank
    n = spec["steps"]
    truth = np.asarray([MODES.index(active_mode(cell, i, spec)) for i in range(n)])
    row = np.arange(n)
    oracle = trace["oracle_observations"]
    np.testing.assert_array_equal(oracle[row, truth], trace["next_states"])
    np.testing.assert_array_equal(trace["oracle_physical_states"][row, truth], trace["actual_physical_states"])
    observation_exact = np.all(oracle == trace["next_states"][:, None, :], axis=2)
    physical_exact = np.all(trace["oracle_physical_states"] == trace["actual_physical_states"][:, None, :], axis=2)
    physical_gaps = np.sqrt(np.mean(((oracle-trace["next_states"][:, None, :])/bank.target_scale)**2, axis=2))
    physical_gaps[row, truth] = np.inf
    nearest_gap = physical_gaps.min(axis=1)
    design = bank.design(trace)
    predictions = np.stack([trace["states"]+(design@bank.coefficients[m]+bank.centers[m])*bank.target_scale
                            for m in MODES], axis=1)
    approximation_errors = np.sqrt(np.mean(((predictions-oracle)/bank.target_scale)**2, axis=2))
    true_model_error = approximation_errors[row, truth]
    pointwise_ll = bank.log_likelihoods(trace)
    observed = observe_series(trace, reference)
    ll = observed["mean_log_likelihoods"]
    other = ll.copy()
    other[row, truth] = -np.inf
    margins = ll[row, truth]-other.max(axis=1)
    predicted = ll.argmax(axis=1)
    z = (bank.features.inputs(trace)-bank.features.input_mean)/bank.features.input_scale
    start = spec["switch_step"]
    windows = dict(first800=(0, min(800, n)), full=(0, n), pre_last64=(max(0, start-64), start),
        first_post50=(start, min(start+50, n)), post50_200=(min(start+50, n-1), min(start+200, n)),
        post=(start, n), last1000=(max(start, n-1000), n), last64=(max(start, n-64), n))
    output = {}
    for name, (left, right) in windows.items():
        if right <= left:
            continue
        s = slice(left, right)
        gap, error = nearest_gap[s], true_model_error[s]
        separated = gap > 0
        ratios = error[separated]/gap[separated]
        output[name] = dict(transitions=right-left,
            observable_unique_true_template_fraction=float(np.mean(observation_exact[s].sum(axis=1) == 1)),
            full_state_unique_true_template_fraction=float(np.mean(physical_exact[s].sum(axis=1) == 1)),
            nearest_other_template={m: float(np.mean(physical_gaps[s].argmin(axis=1) == i)) for i, m in enumerate(MODES)},
            nearest_physical_template_gap_rms_common_units=distribution(gap),
            true_mode_model_error_rms_common_units=distribution(error),
            model_error_over_nearest_template_gap=distribution(ratios) if len(ratios) else None,
            model_error_exceeds_template_gap_fraction=float(np.mean(error > gap)),
            model_vs_oracle_error_rms_by_mode={m: distribution(approximation_errors[s, i]) for i, m in enumerate(MODES)},
            v69_pointwise_mode_accuracy=float(np.mean(pointwise_ll[s].argmax(axis=1) == truth[s])),
            v69_window_mode_accuracy=float(np.mean(predicted[s] == truth[s])),
            v69_mean_true_minus_best_other_log_likelihood=float(margins[s].mean()),
            v69_predicted_mode_fraction={m: float(np.mean(predicted[s] == i)) for i, m in enumerate(MODES)},
            v69_mean_risk=float(observed["risk"][s].mean()),
            input_standardized_max_abs=distribution(np.abs(z[s]).max(axis=1)),
            clipped_input_velocity_fraction=float(trace["clipped_input_velocities"][s].mean()))
    return output


def run(args):
    torch.set_num_threads(2)
    payload = torch.load(args.reference, map_location="cpu", weights_only=False)
    assert payload["revision"] == v69.REVISION and payload["spec"] == v69.SPEC
    reference = payload["reference"]
    metadata = deepcopy(reference.metadata())
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    stored = args.checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir/"preflight_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=SPEC, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == SPEC
    cells = audit_cells(SPEC)
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        env = gym.make("Hopper-v4")
        shadows = v69.make_shadows(env)
        agent = v69.ContinuationAgent(env, "fixed-minus2", args.device)
        agent.start_shared(args.source_dir/v69.SOURCE_NAME, shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            apply_mode(env, active_mode(cell, max(0, loop["step"]-1), SPEC))
            env.reset(seed=cell["seed"])
            v69.restore_step_state(env, extra["environment"])
        trace, record = collect(agent, env, shadows, cell, SPEC, loop,
            lambda active: v69.v68.v53.save_progress(agent, env, progress, active, checkpoint))
        np.savez_compressed(trace_path(stored, cell), **trace)
        if cell["mode"] != "normal":
            normal = v69.v68.load_trace(trace_path(stored, dict(cell, mode="normal")))
            assert v69.v68.v53.paired_prefix_equal(trace, normal, SPEC["switch_step"]), "non-matched-prefix"
        record["windows"] = summarize_trace(trace, cell, reference, SPEC)
        progress["records"].append(record)
        progress["next_cell"] = index+1
        v69.v68.v53.save_progress(agent, env, progress, None, checkpoint)
        for physical in (env, *shadows.values()):
            physical.close()
        extra = None
    records = progress["records"]
    for record in records:
        normal = next(r for r in records if r["seed"] == record["seed"] and r["mode"] == "normal")
        record["paired_risk_contrasts"] = {w: d["v69_mean_risk"]-normal["windows"][w]["v69_mean_risk"]
                                           for w, d in record["windows"].items()}
    assert reference.metadata() == metadata
    totals = {k: sum(r[k] for r in records) for k in ("actual_transitions", "counterfactual_step_calls",
                "environment_step_calls", "exact_true_mode_replays", "optimizer_updates")}
    assert totals["actual_transitions"] == len(cells)*SPEC["steps"] and totals["optimizer_updates"] == 0
    output = dict(revision=REVISION, spec=SPEC, source_training_seed=1, source_updates=99745,
        source_checkpoint=v69.SOURCE_NAME, reference_checkpoint=str(args.reference), reference_metadata=metadata,
        action_sampling="stochastic", episode_time_limit=1000, policy_weights_frozen=True,
        context_updates=True, oracle_controls_actor=False, reference_refitted=False,
        physical_profiles=v69.MUJOCO_MODE_PROFILES["Hopper-v4"], records=records, **totals,
        limitations=["Full-state physical templates and true labels are privileged diagnostics, not an online belief implementation.",
            "Fresh frozen-policy trajectories from one source training seed do not reproduce prior updating-policy failures.",
            "Known mild gravity/compound profiles; neither extreme-OOD performance nor closed-loop return confirmation."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir/"physical_audit.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] Physical audit, no actor updates: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "reference", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
