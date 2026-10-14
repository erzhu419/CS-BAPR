"""Matched reference collection with/without post-switch action exploration."""
import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

REVISION = "general_ood_v104_behavior_exploration"
SEEDS = (135230, 135231)
SMOKE_SEED = 135229
SPEC = dict(seeds=list(SEEDS), steps=6000, switch_step=3800, stride=2,
            action_noise_std=0.15)
BEHAVIORS = ("standard", "exploratory")
COLLECTION_MODES = ("high_grav", "compound")
KEYS = ("states", "executed_actions", "next_states")
COUNTERS = ("actual_transitions", "counterfactual_step_calls", "environment_step_calls",
            "exact_true_mode_replays", "optimizer_updates")
CHECKPOINT_STEPS = 1000


def cells(seed):
    return [dict(role=f"reference_{behavior}", seed=seed, mode=mode, updating=True,
                 group=behavior) for behavior in BEHAVIORS for mode in COLLECTION_MODES]


def reference_action(policy_action, behavior, step, spec):
    if behavior == "standard" or step < spec["switch_step"]:
        return policy_action.copy()
    noise = np.random.normal(0., spec["action_noise_std"], size=policy_action.shape)
    return np.clip(policy_action + noise, -1., 1.).astype(policy_action.dtype)


def collect_updating(worker, agent, env, shadows, cell, spec, loop=None, checkpoint=None):
    assert cell["updating"] and tuple(shadows) == worker.MODES
    if loop is None:
        random.seed(cell["seed"])
        np.random.seed(cell["seed"])
        torch.manual_seed(cell["seed"])
        worker.v70.apply_mode(env, "normal")
        state, _ = env.reset(seed=cell["seed"])
        agent.reset_context()
        loop = dict(step=0, state=state, episode=0, initial_updates=agent.training_steps,
            initial_replay_size=agent.replay_buffer.size, q_loss_sum=0.,
            rows={k: [] for k in (*worker.v68.v53.TRACE_KEYS, *worker.v70.ORACLE_KEYS, "policy_actions")},
            recent={k: deque(maxlen=agent.config.surprise_window)
                    for k in ("state", "action", "reward", "context")})
    for step in range(loop["step"], spec["steps"]):
        mode = worker.v70.active_mode(cell, step, spec)
        if step == spec["switch_step"]:
            worker.v70.apply_mode(env, mode)
        state = loop["state"]
        policy_action = agent.select_action(state, deterministic=False)
        action = reference_action(policy_action, cell["group"], step, spec)
        context = agent._current_context_np().copy()
        executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
        snapshot = worker.v69.environment_state(env)
        clipped = bool(np.any(np.abs(env.unwrapped.data.qvel) > 10.))
        actual = env.step(executed)
        actual_physical = worker.v70.physical_state(env)
        observations, physical = [], []
        for candidate, shadow in shadows.items():
            worker.v69.restore_step_state(shadow, snapshot)
            np.testing.assert_array_equal(shadow.unwrapped._get_obs(), state)
            values = shadow.step(executed)
            next_physical = worker.v70.physical_state(shadow)
            if candidate == mode:
                np.testing.assert_array_equal(values[0], actual[0])
                np.testing.assert_array_equal(next_physical, actual_physical)
                assert values[1:4] == actual[1:4], "true-mode-shadow-not-exact"
            observations.append(values[0].copy())
            physical.append(next_physical)
        worker.v69.append_transition(loop["rows"], state, action, executed, actual)
        loop["rows"]["policy_actions"].append(policy_action.copy())
        for key, value in zip(worker.v70.ORACLE_KEYS, (observations, physical, actual_physical,
                snapshot["values"], snapshot["elapsed_steps"], clipped)):
            loop["rows"][key].append(value)
        next_state, reward, terminated, truncated, _ = actual
        done = terminated or truncated
        for key, value in (("state", state.copy()), ("action", action.copy()),
                           ("reward", float(reward)), ("context", context)):
            loop["recent"][key].append(value)
        # Store the normalized action actually executed, not the actor's pre-noise proposal.
        agent.store_transition(state, action, reward, next_state, float(done))
        metrics = agent.update(recent_rollout=worker.v69._recent_rollout_payload(loop["recent"]))
        assert metrics is not None and metrics["effective_beta"] == -2.
        loop["q_loss_sum"] += float(metrics["q_loss"])
        loop["step"], loop["state"] = step+1, next_state
        if done and step+1 < spec["steps"]:
            loop["episode"] += 1
            loop["state"], _ = env.reset(seed=cell["seed"]+10000*loop["episode"])
            agent.reset_context()
        if checkpoint is not None and (step+1) % CHECKPOINT_STEPS == 0:
            checkpoint(loop)
            print(f"[COLLECT] seed{cell['seed']} {cell['group']} {cell['mode']} {step+1}/{spec['steps']}", flush=True)
    updates = agent.training_steps-loop["initial_updates"]
    assert updates == spec["steps"]
    assert agent.replay_buffer.size == loop["initial_replay_size"]+updates, "oracle-entered-RL-replay"
    profile = worker.v69.MUJOCO_MODE_PROFILES["Hopper-v4"][cell["mode"]]
    raw = env.unwrapped.model
    np.testing.assert_array_equal(raw.body_mass, env._original_body_mass*profile["body_mass"])
    np.testing.assert_array_equal(raw.geom_friction, env._original_friction*profile["friction"])
    np.testing.assert_array_equal(raw.opt.gravity, env._original_gravity*profile["gravity"])
    trace = {k: np.asarray(v) for k, v in loop["rows"].items()}
    suffix = slice(spec["switch_step"], spec["steps"])
    delta = trace["actions"][suffix]-trace["policy_actions"][suffix]
    record = dict(**cell, actual_transitions=spec["steps"], counterfactual_step_calls=5*spec["steps"],
        environment_step_calls=6*spec["steps"], exact_true_mode_replays=spec["steps"], optimizer_updates=updates,
        physical_terminations=int(trace["terminated"].sum()), time_limit_truncations=int(trace["truncated"].sum()),
        clipped_input_velocities=int(trace["clipped_input_velocities"].sum()), mean_q_loss=loop["q_loss_sum"]/updates,
        post_switch_action_delta_rms=float(np.sqrt(np.mean(delta**2))),
        post_switch_action_at_bound_fraction=float(np.mean(np.abs(trace["actions"][suffix]) == 1.)))
    return trace, record


def collect(args, source, spec):
    import gymnasium as gym

    worker = source.v76
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, seed=args.seed, spec=spec, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["seed"] == args.seed and progress["spec"] == spec
    population = cells(args.seed)
    assert progress["next_cell"] == len(progress["records"])
    assert [(r["group"], r["mode"]) for r in progress["records"]] == [
        (c["group"], c["mode"]) for c in population[:progress["next_cell"]]]
    stored = args.checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir/"latest.pt"
    for i in range(progress["next_cell"], len(population)):
        cell = population[i]
        env = gym.make("Hopper-v4")
        shadows = worker.v69.make_shadows(env)
        agent = worker.v69.ContinuationAgent(env, "fixed-minus2", args.device)
        agent.start_shared(args.source_dir/worker.v69.SOURCE_NAME, shared)
        assert (agent.training_steps, agent.context_tracker._count, agent.replay_buffer.size) == (99745, 100000, 6144)
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            worker.v70.apply_mode(env, worker.v70.active_mode(cell, max(0, loop["step"]-1), spec))
            env.reset(seed=cell["seed"])
            worker.v69.restore_step_state(env, extra["environment"])
        trace, record = collect_updating(worker, agent, env, shadows, cell, spec, loop,
            lambda active: worker.v68.v53.save_progress(agent, env, progress, active, checkpoint))
        np.savez_compressed(worker.trace_path(stored, cell), **trace)
        if i:
            first = worker.v68.load_trace(worker.trace_path(stored, population[0]))
            switch = spec["switch_step"]
            assert all(np.array_equal(trace[k][:switch], first[k][:switch]) for k in trace), "non-matched-prefix"
            for key in ("states", "policy_actions"):
                np.testing.assert_array_equal(trace[key][switch], first[key][switch])
        progress["records"].append(record)
        progress["next_cell"] = i+1
        worker.v68.v53.save_progress(agent, env, progress, None, checkpoint)
        for physical in (env, *shadows.values()):
            physical.close()
        del agent
        torch.cuda.empty_cache()
        extra = None
        print(f"[PATH] {i+1}/4 seed{args.seed} {cell['group']} {cell['mode']}", flush=True)
    return progress["records"]


def extensions(traces, spec, modes):
    indices = np.arange(spec["switch_step"], spec["steps"], spec["stride"])
    output, sources = {}, {}
    for behavior in BEHAVIORS:
        selected = [traces[behavior, m] for m in COLLECTION_MODES]
        output[behavior] = {mode: {key: np.concatenate([
            t["oracle_observations"][indices, i] if key == "next_states" else t[key][indices]
            for t in selected]) for key in KEYS} for i, mode in enumerate(modes)}
        sources[behavior] = [dict(source_mode=m, behavior=behavior,
            source_steps=[spec["switch_step"], spec["steps"]], stride=spec["stride"],
            input_row_range=[j*len(indices), (j+1)*len(indices)]) for j, m in enumerate(COLLECTION_MODES)]
        for m, trace in zip(COLLECTION_MODES, selected):
            np.testing.assert_array_equal(trace["oracle_observations"][indices, modes.index(m)],
                                          trace["next_states"][indices])
        for mode in modes:
            for key in ("states", "executed_actions"):
                np.testing.assert_array_equal(output[behavior][mode][key], output[behavior]["normal"][key])
    assert len(output["standard"]["normal"]["states"]) == len(output["exploratory"]["normal"]["states"])
    return output, sources


def check_seed(args, source):
    assert args.seed == SMOKE_SEED if args.smoke else args.seed in SEEDS
    selection = json.loads((args.v83_dir/"run_checkpoints/cross-seed-covariance/nested_selection.json").read_text())
    used = {b["seed"] for b in selection["reference_blocks"]}
    used.update(source.v76.SPEC["test"])
    used.update((135199, 135200, 135201))
    for version, filename in (("v94", "suffix_replay.json"), ("v95", "fresh_seed.json")):
        directory = getattr(args, f"{version}_dir")
        paths = [directory/"run_results/replay"/filename] if version == "v94" else [
            directory/f"run_results/seed{seed}"/filename for seed in (135190, 135191, 135192)]
        for path in paths:
            used.update(r["seed"] for r in json.loads(path.read_text())["records"])
    used.update(source.v76.v69.SPEC[k] for k in ("fit_seed", "density_calibration_seed", "normal_null_seed"))
    assert not (set(SEEDS) | {SMOKE_SEED}) & used, "reference seed overlaps existing reference/evaluation"


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    check_seed(args, source)
    spec = dict(SPEC, seeds=[SMOKE_SEED], steps=64, switch_step=32) if args.smoke else SPEC
    records = collect(args, source, spec)
    traces = {(c["group"], c["mode"]): source.v76.v68.load_trace(
        source.v76.trace_path(args.checkpoint_dir/"traces", c)) for c in cells(args.seed)}
    payload, sources = extensions(traces, spec, source.MODES)
    counts = {behavior: len(data["normal"]["states"]) for behavior, data in payload.items()}
    assert all(n == (32 if args.smoke else 2200) for n in counts.values())
    totals = {k: sum(r[k] for r in records) for k in COUNTERS}
    assert totals["actual_transitions"] == totals["optimizer_updates"] == 4*spec["steps"]
    budgets = {behavior: {k: sum(r[k] for r in records if r["group"] == behavior) for k in COUNTERS}
               for behavior in BEHAVIORS}
    assert budgets["standard"] == budgets["exploratory"]
    for behavior in sources:
        for row in sources[behavior]:
            row.update(seed=args.seed, role=f"reference_{behavior}", updating=True)
    torch.save(dict(revision=REVISION, seed=args.seed, spec=spec, extensions=payload, sources=sources,
        groups={c: np.full(n, args.seed, dtype=np.int64) for c, n in counts.items()}),
        args.checkpoint_dir/"reference_inputs.pt")
    result = dict(revision=REVISION, stage="reference-collection", passed=True, seed=args.seed, spec=spec,
        smoke=args.smoke, collection_device=args.device, source_training_seed=1, source_updates=99745,
        behaviors=list(BEHAVIORS), collection_modes=list(COLLECTION_MODES), collection_records=records,
        **totals, per_behavior_budgets=budgets, per_behavior_reference_inputs=counts,
        per_behavior_counterfactual_labels={c: n*5 for c, n in counts.items()}, sources=sources,
        paired_prefixes=True, equal_behavior_budgets=True, exact_true_mode_replays_verified=True,
        actual_actions_enter_replay_and_context=True, counterfactuals_added_to_RL_replay=False,
        evaluation_paths_used_for_reference=False, reference_banks_fitted=False,
        normal_thresholds_recalibrated=False, coefficient_tuned=False,
        readout_controls_actor=False, readout_adopted=False,
        decision="collection-only-pending-mean-support-pilot",
        limitations=["Known-mode simulator counterfactual labels construct offline references.",
            "Collection seeds share one trained source policy; they are not independent training seeds.",
            "This is a development coverage pilot, not evidence of recognition or return improvement."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "reference_collection.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v83-dir", "v94-dir", "v95-dir", "source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
