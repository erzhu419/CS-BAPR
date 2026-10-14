"""Matched-prefix second switches, with frozen policies and normal-only references."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v41 as v41
from scripts import probe_general_ood_v42 as v42
from scripts import probe_general_ood_v43 as v43
from scripts import probe_general_ood_v44 as v44
from scripts import probe_general_ood_v46 as v46
from scripts import run_general_ood_v40 as v40
from scripts import v44_belief as belief
from scripts.train_csbapr import MUJOCO_MODE_PROFILES, apply_mode_to_env
from scripts.v46_belief import RobustAdaptiveStudent

REVISION = "general_ood_v47_multiple_switch"
FIRST_SWITCH, SECOND_SWITCH, PROBE_STEPS = 1024, 1280, 1792
INITIAL_MODES = ("heavy", "slippery")
ARM_CALIBRATIONS = {"episode-original": "episode-centered", "global-robust": "global-centered"}
PHASES = dict(v46.PHASES, prefix1280=(0, 1280), second_early16=(1280, 1296),
              second_early64=(1280, 1344), second_all512=(1280, 1792),
              second_late256=(1536, 1792), second_last64=(1728, 1792), full1792=(0, 1792))


@torch.no_grad()
def collect_trajectory(agent, env, seed, switches, snapshot, budget=PROBE_STEPS):
    agent.restore_context(deepcopy(snapshot))
    agent.reset_context()
    profiles = MUJOCO_MODE_PROFILES["Hopper-v4"]
    apply_mode_to_env(env, "Hopper-v4", profiles["normal"])
    state, _ = env.reset(seed=seed)
    episodes, rows = 0, {k: [] for k in ("states", "actions", "executed_actions", "next_states",
                                        "contexts", "rewards", "dones", "terminated", "truncated")}
    for step in range(budget):
        if step in switches:
            apply_mode_to_env(env, "Hopper-v4", profiles[switches[step]])
        action = agent.select_action(state, deterministic=True)
        context = agent._current_context_np()
        executed = np.clip(action * env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(executed)
        done = terminated or truncated
        for key, value in (("states", state.copy()), ("actions", action.copy()),
                           ("executed_actions", executed.copy()), ("next_states", next_state.copy()),
                           ("contexts", context.copy()), ("rewards", float(reward)),
                           ("dones", bool(done)), ("terminated", bool(terminated)), ("truncated", bool(truncated))):
            rows[key].append(value)
        agent.observe_context(state, action, reward, next_state, done=done)
        state = next_state
        if done and step + 1 < budget:
            episodes += 1
            state, _ = env.reset(seed=seed + episodes * 10000)
            agent.reset_context()
    trace = {k: np.asarray(v) for k, v in rows.items()}
    spread = []
    for start in range(0, budget, v41.WINDOW):
        states = torch.as_tensor(trace["states"][start:start + v41.WINDOW], dtype=torch.float32, device=agent.device)
        contexts = torch.as_tensor(trace["contexts"][start:start + v41.WINDOW], dtype=torch.float32, device=agent.device)
        actions = torch.as_tensor(trace["actions"][start:start + v41.WINDOW], dtype=torch.float32, device=agent.device)
        spread.extend(agent.critic(agent._augment_state_tensor(states, contexts), actions).std(dim=0).cpu().numpy().tolist())
    trace["q_spread"] = np.asarray(spread, dtype=np.float32)
    return trace


def observe(trace, model):
    ages, errors = v43.episode_steps(trace["dones"]), model.errors(trace)
    arms, histories, signals = {}, {}, {}
    for name, channel in ARM_CALIBRATIONS.items():
        calibration = model.calibrations[channel]
        scores, thresholds = calibration.scores(errors, ages)
        signals[channel] = signal = np.log1p(scores / thresholds)
        tracker = (RobustAdaptiveStudent if name == "global-robust" else belief.AdaptiveStudent)(**belief.SETTINGS)
        metrics, events, groups = [], [], calibration.groups(ages)
        for step, value in enumerate(signal):
            tracker.update(float(value))
            penalty = float(tracker.belief @ np.exp(-.1 * np.arange(tracker.max_H)))
            metrics.append((tracker.belief[0], tracker.effective_window, penalty, -2. - penalty, tracker.entropy))
            if tracker.belief[0] > .5:
                events.append(dict(step=step, episode_age=int(ages[step]), signal=float(value),
                                   reset_probability=float(tracker.belief[0]),
                                   calibration_changed=bool(step and groups[step] != groups[step - 1]),
                                   previous_transition_ended=bool(step and trace["dones"][step - 1]),
                                   terminated=bool(trace["terminated"][step]), truncated=bool(trace["truncated"][step])))
        histories[name] = metrics = np.asarray(metrics)
        arms[name] = dict(phases={p: v44.phase_statistics(metrics[a:b]) for p, (a, b) in PHASES.items()},
                          major_reset_transition_counts={p: int(np.sum(metrics[a:b, 0] > .5)) for p, (a, b) in PHASES.items()},
                          major_reset_events=events)
    checks, _ = v43.observe_trace(trace, {v44.PREDICTOR: model})
    return arms, histories, signals, checks[v44.PREDICTOR][v44.CALIBRATOR]


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    fit_seeds, calibration_seeds, seeds = v43.stage_seeds(args.stage)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    for policy in v41.POLICIES:
        started = time.monotonic()
        fresh = gym.make("Hopper-v4")
        agent = v40.make_agent("Hopper-v4", policy, "cuda", fresh)
        fresh.close()
        checkpoint = args.checkpoint_dir / f"{v40.REVISION}_Hopper-v4_{policy}_seed1_final.pt"
        agent.load(str(checkpoint))
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000
        for network in (agent.actor, agent.critic, agent.target_critic):
            network.eval()
            network.requires_grad_(False)
        snapshot, references, datasets = agent.snapshot_context(), [], {}
        for split, ref_seeds in (("fit", fit_seeds), ("calibration", calibration_seeds)):
            traces = []
            for seed in ref_seeds:
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v41.collect_trajectory(agent, env, seed, "normal", snapshot, budget=v43.REFERENCE_STEPS)
                env.close()
                traces.append(trace)
                references.append(v43.reference_record(trace, seed, split))
            datasets[split] = v43.concatenate_references(traces)
        model = v43.ReferenceDynamics(*v42.PREDICTORS[v44.PREDICTOR]).fit_references(datasets["fit"], datasets["calibration"])
        metadata = model.metadata()
        del datasets, traces, trace
        print(f"[REFERENCE] {policy}: unchanged normal-only fit/calibration", flush=True)
        records = []
        for seed in seeds:
            for initial in INITIAL_MODES:
                other = next(m for m in INITIAL_MODES if m != initial)
                for target in (initial, "normal", other):
                    switches = {FIRST_SWITCH: initial}
                    if target != initial:
                        switches[SECOND_SWITCH] = target
                    env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                    trace = collect_trajectory(agent, env, seed, switches, snapshot)
                    env.close()
                    arms, histories, signals, check = observe(trace, model)
                    if target == initial:
                        control, control_histories, control_signals = trace, histories, signals
                    assert v41.prefix_equal(control, trace, SECOND_SWITCH), (policy, seed, initial, target, "unpaired-prefix")
                    assert all(np.array_equal(control[k][SECOND_SWITCH], trace[k][SECOND_SWITCH])
                               for k in ("states", "executed_actions", "contexts")), "unpaired-switch-inputs"
                    assert all(np.array_equal(control_histories[k][:SECOND_SWITCH], histories[k][:SECOND_SWITCH]) for k in arms)
                    assert all(np.array_equal(control_signals[k][:SECOND_SWITCH], signals[k][:SECOND_SWITCH]) for k in signals)
                    records.append(dict(seed=seed, initial_mode=initial, target_mode=target, actual_second_change=target != initial,
                                        paired_prefix_equal=True, paired_switch_inputs_equal=True, paired_belief_prefix_equal=True,
                                        interactions=len(trace["dones"]), arms=arms, v43_score_check=check,
                                        signal={c: {p: float(s[a:b].mean()) for p, (a, b) in PHASES.items()} for c, s in signals.items()},
                                        phase_endings={p: {k: int(trace[k][a:b].sum()) for k in ("terminated", "truncated")}
                                                       for p, (a, b) in PHASES.items()},
                                        reward_means_by_256=[float(trace["rewards"][i:i + 256].mean()) for i in range(0, PROBE_STEPS, 256)],
                                        termination_counts_by_256=[int(trace["terminated"][i:i + 256].sum()) for i in range(0, PROBE_STEPS, 256)],
                                        truncation_counts_by_256=[int(trace["truncated"][i:i + 256].sum()) for i in range(0, PROBE_STEPS, 256)]))
            print(f"[MULTISWITCH] {policy} seed={seed}: four changed + two stay branches", flush=True)
        assert model.metadata() == metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = dict(revision=REVISION, stage=args.stage, source_policy=policy, source_checkpoint=checkpoint.name,
                      source_training_seed=1, config=asdict(agent.config), references=references, model_metadata=metadata,
                      fit_seeds=fit_seeds, calibration_seeds=calibration_seeds, evaluation_seeds=seeds,
                      first_switch=FIRST_SWITCH, second_switch=SECOND_SWITCH, probe_steps=PROBE_STEPS,
                      episode_time_limit=v41.PROBE_STEPS, initial_modes=INITIAL_MODES, arm_calibrations=ARM_CALIBRATIONS,
                      filter_settings=belief.SETTINGS, robust_student_degrees=4, phases=PHASES,
                      optimizer_updates_during_probe=0, replay_insertions=0, auxiliary_signal_controls_actions=False,
                      reset_filters_at_episode_end=False, predictors_updated_after_calibration=False, actor_critic_target_frozen=True,
                      loaded_training_update_steps=agent.training_steps, loaded_training_context_count=agent.context_tracker._count,
                      records=records, total_reference_interactions=len(references) * v43.REFERENCE_STEPS,
                      total_probe_interactions=len(records) * PROBE_STEPS, seconds=time.monotonic() - started,
                      runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
                                   mujoco=__import__("mujoco").__version__, gpu=torch.cuda.get_device_name()))
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"[DONE] multiple switches {policy}: result={path}", flush=True)
        del agent, model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
