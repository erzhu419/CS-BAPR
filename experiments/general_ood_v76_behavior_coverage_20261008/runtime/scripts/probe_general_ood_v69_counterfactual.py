"""Matched simulator-state mode references with frozen/updating normal coverage."""
import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import mujoco
import numpy as np
import torch

from scripts import probe_general_ood_v68_mode_likelihood as v68
from scripts.run_general_ood_v52 import SOURCE_NAME, environment_state
from scripts.train_csbapr import MUJOCO_MODE_PROFILES, apply_mode_to_env, _recent_rollout_payload
from scripts.v52_agent import ContinuationAgent
from scripts.v68_mode_likelihood import MODES, NAME, ModeLikelihoodReference

REVISION = "general_ood_v69_counterfactual"
SPEC = dict(fit_seed=135040, density_calibration_seed=135050, normal_null_seed=135060, steps=6000)
CHECKPOINT_STEPS = 1000
PREVIOUS = "v68_separate_mode"


def reference_cells(spec):
    seeds = (spec["fit_seed"], spec["density_calibration_seed"], spec["normal_null_seed"])
    assert len(set(seeds)) == 3
    return [dict(role=role, seed=seed, updating=updating) for role, seed in zip(
        ("fit", "density_calibration", "normal_null"), seeds) for updating in (False, True)]


def trace_path(directory, cell, mode="source_normal"):
    kind = "updating" if cell["updating"] else "frozen"
    return directory/f"{cell['role']}_seed{cell['seed']}_{kind}_{mode}.npz"


def make_shadows(source):
    shadows = {}
    for mode in MODES:
        env = gym.make("Hopper-v4", max_episode_steps=source._max_episode_steps)
        apply_mode_to_env(env, "Hopper-v4", MUJOCO_MODE_PROFILES["Hopper-v4"][mode])
        env.reset(seed=0)
        shadows[mode] = env
    return shadows


def restore_step_state(env, snapshot):
    raw = env.unwrapped
    # mj_step recomputes dynamics; an extra mj_forward alters solver warm starts.
    mujoco.mj_setState(raw.model, raw.data, snapshot["values"], mujoco.mjtState.mjSTATE_INTEGRATION)
    raw.np_random.bit_generator.state = snapshot["random_state"]
    env._elapsed_steps = snapshot["elapsed_steps"]


def shadow_steps(shadows, snapshot, state, executed, actual):
    output = {}
    for mode, env in shadows.items():
        restore_step_state(env, snapshot)
        np.testing.assert_array_equal(env.unwrapped._get_obs(), state)
        values = env.step(executed)
        if mode == "normal":
            np.testing.assert_array_equal(values[0], actual[0])
            assert values[1:4] == actual[1:4], "normal-counterfactual-not-exact"
        output[mode] = values
    return output


def append_transition(rows, state, action, executed, values):
    next_state, reward, terminated, truncated, _ = values
    data = (state.copy(), action.copy(), executed.copy(), next_state.copy(), float(reward),
            bool(terminated or truncated), bool(terminated), bool(truncated))
    for key, value in zip(v68.v53.TRACE_KEYS, data):
        rows[key].append(value)


def collect_reference(agent, env, shadows, cell, steps, loop=None, checkpoint=None):
    assert tuple(shadows) == (MODES if cell["role"] != "normal_null" else ())
    apply_mode_to_env(env, "Hopper-v4", MUJOCO_MODE_PROFILES["Hopper-v4"]["normal"])
    if loop is None:
        random.seed(cell["seed"])
        np.random.seed(cell["seed"])
        torch.manual_seed(cell["seed"])
        state, _ = env.reset(seed=cell["seed"])
        agent.reset_context()
        loop = dict(step=0, state=state, episode=0, initial_updates=agent.training_steps,
            initial_replay_size=agent.replay_buffer.size, clipped_velocity_transitions=0,
            rows={k: [] for k in v68.v53.TRACE_KEYS},
            counterfactual_rows={m: {k: [] for k in v68.v53.TRACE_KEYS} for m in shadows},
            recent={k: deque(maxlen=agent.config.surprise_window) for k in ("state", "action", "reward", "context")},
            q_loss_sum=0.)
    for step in range(loop["step"], steps):
        state = loop["state"]
        action = agent.select_action(state, deterministic=False)
        context = agent._current_context_np().copy()
        executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
        snapshot = environment_state(env) if shadows else None
        loop["clipped_velocity_transitions"] += int(np.any(np.abs(env.unwrapped.data.qvel) > 10.))
        actual = env.step(executed)
        for mode, values in shadow_steps(shadows, snapshot, state, executed, actual).items():
            append_transition(loop["counterfactual_rows"][mode], state, action, executed, values)
        append_transition(loop["rows"], state, action, executed, actual)
        next_state, reward, terminated, truncated, _ = actual
        done = terminated or truncated
        if cell["updating"]:
            for key, value in (("state", state.copy()), ("action", action.copy()), ("reward", float(reward)), ("context", context)):
                loop["recent"][key].append(value)
            agent.store_transition(state, action, reward, next_state, float(done))
            metrics = agent.update(recent_rollout=_recent_rollout_payload(loop["recent"]))
            assert metrics is not None and metrics["effective_beta"] == -2.
            loop["q_loss_sum"] += float(metrics["q_loss"])
        else:
            agent.observe_context(state, action, reward, next_state, done)
        loop["step"], loop["state"] = step+1, next_state
        if done and step+1 < steps:
            loop["episode"] += 1
            loop["state"], _ = env.reset(seed=cell["seed"]+10000*loop["episode"])
            agent.reset_context()
        if checkpoint is not None and (step+1) % CHECKPOINT_STEPS == 0:
            checkpoint(loop)
            print(f"[COLLECT] {cell['role']} seed{cell['seed']} updating={cell['updating']} {step+1}/{steps}", flush=True)
    updates = agent.training_steps-loop["initial_updates"]
    assert updates == (steps if cell["updating"] else 0)
    assert agent.replay_buffer.size == loop["initial_replay_size"]+updates, "counterfactual-entered-RL-replay"
    for mode, physical in (("normal", env), *shadows.items()):
        profile = MUJOCO_MODE_PROFILES["Hopper-v4"][mode]
        raw = physical.unwrapped.model
        np.testing.assert_array_equal(raw.body_mass, physical._original_body_mass*profile["body_mass"])
        np.testing.assert_array_equal(raw.geom_friction, physical._original_friction*profile["friction"])
        np.testing.assert_array_equal(raw.opt.gravity, physical._original_gravity*profile["gravity"])
    normal = {k: np.asarray(v) for k, v in loop["rows"].items()}
    counterfactuals = {m: {k: np.asarray(v) for k, v in rows.items()} for m, rows in loop["counterfactual_rows"].items()}
    record = dict(**cell, normal_interactions=steps, counterfactual_step_calls=steps*len(shadows),
        environment_interactions=steps*(1+len(shadows)), labelled_mode_transitions=steps*len(shadows),
        exact_normal_replay_checks=steps if shadows else 0, optimizer_updates=updates,
        physical_terminations=int(normal["terminated"].sum()), time_limit_truncations=int(normal["truncated"].sum()),
        clipped_velocity_transitions=loop["clipped_velocity_transitions"],
        mean_q_loss=loop["q_loss_sum"]/updates if updates else None)
    return normal, counterfactuals, record


def fit_reference(stored, cells):
    datasets = {}
    for role in ("fit", "density_calibration"):
        streams = [c for c in cells if c["role"] == role]
        datasets[role] = {}
        for mode in MODES:
            traces = [v68.load_trace(trace_path(stored, c, mode)) for c in streams]
            datasets[role][mode] = {k: np.concatenate([t[k] for t in traces]) for k in v68.v53.TRACE_KEYS}
        for mode in MODES[1:]:
            for key in ("states", "actions", "executed_actions"):
                np.testing.assert_array_equal(datasets[role][mode][key], datasets[role]["normal"][key])
    nulls = [v68.load_trace(trace_path(stored, c)) for c in cells if c["role"] == "normal_null"]
    return ModeLikelihoodReference.fit(datasets["fit"], datasets["density_calibration"], nulls)


def run(args):
    torch.set_num_threads(2)
    cells = reference_cells(SPEC)
    stored = args.checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir/"preflight_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=SPEC, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == SPEC
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        env = gym.make("Hopper-v4")
        shadows = make_shadows(env) if cell["role"] != "normal_null" else {}
        agent = ContinuationAgent(env, "fixed-minus2", args.device)
        agent.start_shared(args.source_dir/SOURCE_NAME, shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            apply_mode_to_env(env, "Hopper-v4", MUJOCO_MODE_PROFILES["Hopper-v4"]["normal"])
            env.reset(seed=cell["seed"])
            restore_step_state(env, extra["environment"])
        normal, counterfactuals, record = collect_reference(agent, env, shadows, cell, SPEC["steps"], loop,
            lambda active: v68.v53.save_progress(agent, env, progress, active, checkpoint))
        np.savez_compressed(trace_path(stored, cell), **normal)
        for mode, trace in counterfactuals.items():
            np.savez_compressed(trace_path(stored, cell, mode), **trace)
        progress["records"].append(record)
        progress["next_cell"] = index+1
        v68.v53.save_progress(agent, env, progress, None, checkpoint)
        for physical in (env, *shadows.values()):
            physical.close()
        del agent
        torch.cuda.empty_cache()
        extra = None
    reference = fit_reference(stored, cells)
    metadata = deepcopy(reference.metadata())
    path = args.checkpoint_dir/"counterfactual_reference.pt"
    torch.save(dict(revision=REVISION, reference=reference, reference_metadata=metadata, spec=SPEC,
                    source_training_seed=1, source_checkpoint=SOURCE_NAME), path)
    print("[REFERENCE] Matched-state models frozen before replay; likelihood/readout settings unchanged", flush=True)
    baseline = json.loads((args.v62_dir/"run_results/preflight/causal_innovation.json").read_text())
    previous = json.loads((args.v68_dir/"run_results/screen/mode_likelihood.json").read_text())
    assert baseline["revision"] == v68.BASE.REVISION and baseline["stage_spec"] == v68.BASE.SPEC
    assert previous["revision"] == v68.REVISION and len(previous["records"]) == 17
    assert not {c["seed"] for c in cells} & {r["seed"] for r in previous["records"]}
    records = v68.replay_cells(args, reference, previous, baseline)
    for new, old in zip(records, previous["records"]):
        assert all(new[k] == old[k] for k in ("group", "seed", "mode", "updating"))
        new["observers"][PREVIOUS] = deepcopy(old["observers"][NAME])
        if new["updating"]:
            new["change_minus_nochange"][PREVIOUS] = deepcopy(old["change_minus_nochange"][NAME])
    summary = v68.summarize(records)
    for group in summary:
        summary[group][PREVIOUS] = deepcopy(previous["summary"][group][NAME])
    assert reference.metadata() == metadata
    detection_pass = all(s[NAME]["detection_gate_pass"] for s in summary.values())
    identification_pass = all(all(n == s[NAME]["changed_pairs"] for n in s[NAME]["positive_true_mode_margin_pairs"].values())
                              for s in summary.values())
    totals = {k: sum(r[k] for r in progress["records"]) for k in (
        "normal_interactions", "counterfactual_step_calls", "environment_interactions", "labelled_mode_transitions",
        "exact_normal_replay_checks", "optimizer_updates")}
    assert totals == dict(normal_interactions=6*SPEC["steps"], counterfactual_step_calls=20*SPEC["steps"],
        environment_interactions=26*SPEC["steps"], labelled_mode_transitions=20*SPEC["steps"],
        exact_normal_replay_checks=4*SPEC["steps"], optimizer_updates=3*SPEC["steps"])
    output = dict(revision=REVISION, reference_spec=SPEC, source_training_seed=1, source_checkpoint=SOURCE_NAME,
        reference_records=progress["records"], **totals, source_updates=99745, action_sampling="stochastic",
        episode_time_limit=1000, beta=-2., readout_controls_actor=False, counterfactuals_added_to_RL_replay=False,
        known_mode_prior="five labelled simulator profiles and full-state counterfactual simulation access",
        physical_profiles=MUJOCO_MODE_PROFILES["Hopper-v4"], reference_metadata=metadata,
        normal_null_streams_separate=True, source_checkpoint_modified=False, reference_frozen_during_replay=True,
        server_reference_checkpoint=str(path), records=records, summary=summary,
        replayed_saved_transitions=sum(r["saved_transitions"] for r in records),
        detection_gate_pass=detection_pass, true_mode_margin_gate_pass=identification_pass,
        decision="retrospective-counterfactual-screen-pass-not-confirmation" if detection_pass and identification_pass else "hold-counterfactual-screen",
        limitations=["Known simulator profiles and full integration-state access are additional observer information.",
            "Changes both matched-state sampling and normal updating-policy coverage; not an isolated single-factor comparison with v68.",
            "Normal-path states do not cover every later shifted-state trajectory or remove clipped-observation aliasing.",
            "Reused development traces from one source training seed; reference-branch updates are not adaptive-control return evidence."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir/"counterfactual.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "v61-dir", "v62-dir", "v68-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
