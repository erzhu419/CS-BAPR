"""Four-arm matched warm continuation; large state remains on the worker."""
from __future__ import annotations

import argparse
from collections import deque
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import mujoco
import numpy as np
import torch
from tqdm import tqdm

from csbapr.belief.departure import NormalAgeReference, SETTINGS as READOUT_SETTINGS
from csbapr.envs.extreme_ood import ExtremeOODConfig, ExtremeOODWrapper
from csbapr.utils import ReplayBuffer
from scripts import probe_general_ood_v43 as v43
from scripts import probe_general_ood_v49 as v49
from scripts import run_general_ood_v40 as v40
from scripts.train_csbapr import MUJOCO_MODE_PROFILES, apply_mode_to_env, _evaluate_agent, _recent_rollout_payload
from scripts.v52_agent import ContinuationAgent, NormalDynamics, TransitionReadouts, VARIANTS

REVISION = "general_ood_v52_matched_continuation"
BUDGETS = dict(preflight=6000, formal=30000)
REFERENCE_STEPS = 3072
SOURCE_NAME = f"{v40.REVISION}_Hopper-v4_fixed-minus2_seed1_final.pt"
METRICS = ("q_loss", "policy_loss", "q_std", "alpha", "effective_beta", "weighted_lambda_effective",
           "signal", "departure_coefficient", "age_coefficient")


def stage_spec(stage):
    if stage == "preflight":
        return dict(fit=list(range(97920, 97922)), calibration=list(range(97930, 97932)),
                    null=list(range(97940, 97942)), evaluation_start=99520, episodes=2, fixed_episodes=1)
    return dict(fit=list(range(98100, 98108)), calibration=list(range(98110, 98114)),
                null=list(range(98120, 98124)), evaluation_start=91500, episodes=20, fixed_episodes=5)


@torch.no_grad()
def collect_normal(agent, seed, snapshot):
    agent.restore_context(deepcopy(snapshot))
    agent.reset_context()
    torch.manual_seed(seed)
    env = gym.make("Hopper-v4")
    state, _ = env.reset(seed=seed)
    episodes, rows = 0, {k: [] for k in ("states", "actions", "executed_actions", "next_states",
        "contexts", "next_contexts", "rewards", "dones", "terminated", "truncated")}
    for step in range(REFERENCE_STEPS):
        action = agent.select_action(state, deterministic=False)
        context = agent._current_context_np().copy()
        executed = np.clip(action * env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(executed)
        done = terminated or truncated
        agent.observe_context(state, action, reward, next_state, done)
        for k, v in (("states", state.copy()), ("actions", action.copy()), ("executed_actions", executed.copy()),
            ("next_states", next_state.copy()), ("contexts", context), ("next_contexts", agent._current_context_np().copy()),
            ("rewards", float(reward)), ("dones", bool(done)), ("terminated", bool(terminated)), ("truncated", bool(truncated))):
            rows[k].append(v)
        state = next_state
        if done and step + 1 < REFERENCE_STEPS:
            episodes += 1
            state, _ = env.reset(seed=seed + episodes * 10000)
            agent.reset_context()
    env.close()
    return {k: np.asarray(v) for k, v in rows.items()}


def prepare_shared(source, destination, stage):
    env = gym.make("Hopper-v4")
    agent = ContinuationAgent(env, "fixed-minus2")
    env.close()
    agent.load(str(source))
    assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 0
    snapshot, spec, datasets, records = agent.snapshot_context(), stage_spec(stage), {}, []
    replay = ReplayBuffer(agent.config.buffer_size)
    for split in ("fit", "calibration"):
        traces = []
        for seed in spec[split]:
            trace = collect_normal(agent, seed, snapshot)
            records.append(v43.reference_record(trace, seed, split))
            traces.append(trace)
            if split == "fit":
                for i in range(REFERENCE_STEPS):
                    replay.push(trace["states"][i], trace["actions"][i], trace["rewards"][i],
                        trace["next_states"][i], float(trace["dones"][i]), trace["contexts"][i], trace["next_contexts"][i])
            print(f"[NORMAL] {stage} {split} seed={seed}: stochastic actions", flush=True)
        datasets[split] = v43.concatenate_references(traces)
    model = NormalDynamics(2, True).fit_references(datasets["fit"], datasets["calibration"])
    calibration = datasets["calibration"]
    scores, thresholds = model.calibrations["global-centered"].scores(model.errors(calibration), calibration["episode_steps"])
    reference = NormalAgeReference.fit_normal(np.log1p(scores / thresholds), calibration["episode_steps"])
    noise, noise_metadata = v49.fit_startup_noise(model, calibration)
    nulls = []
    for seed in spec["null"]:
        trace = collect_normal(agent, seed, snapshot)
        observer, coefficients, starts = TransitionReadouts(model, reference, noise), [], []
        ages = v43.episode_steps(trace["dones"])
        for i, age in enumerate(ages):
            observer.observe(trace["states"][i], trace["executed_actions"][i], trace["next_states"][i], int(age))
            coefficients.append(observer.departure.risk)
            if age == 0:
                starts.append(i)
        windows = [float(np.mean(coefficients[s:s + 50])) for s in starts if s + 50 <= len(coefficients)]
        nulls.append(dict(seed=seed, mean_coefficient=float(np.mean(coefficients)), peak_coefficient=max(coefficients),
                          complete_start50_means=windows))
    metadata = dict(revision=REVISION, stage=stage, source_checkpoint=source.name,
        source_training_seed=1, loaded_training_update_steps=99745, loaded_context_count=100000,
        action_sampling="stochastic", episode_time_limit=1000, stage_spec=spec,
        normal_records=records, fit_calibration_interactions=len(records) * REFERENCE_STEPS,
        null_interactions=len(nulls) * REFERENCE_STEPS, replay_size=replay.size, replay_split="fit-only",
        model_metadata=model.metadata(), risk_metadata=reference.metadata(), readout_settings=READOUT_SETTINGS,
        noise_metadata=noise_metadata, null_records=nulls, reference_frozen=True,
        initial_reward_ema_var=1., resume_kind="matched-warm-continuation-not-exact-v40-resume")
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(dict(replay=replay, context=snapshot, model=model, reference=reference, startup_noise=noise,
                    metadata=metadata), destination / "shared.tmp")
    (destination / "shared.tmp").replace(destination / "shared.pt")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"[SHARED] {stage}: replay={replay.size}, normal starts={reference.normal_start_count}", flush=True)


def environment_state(env):
    raw = env.unwrapped
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(raw.model, spec))
    mujoco.mj_getState(raw.model, raw.data, values, spec)
    return dict(values=values, random_state=deepcopy(raw.np_random.bit_generator.state), elapsed_steps=env._elapsed_steps)


def restore_environment(env, payload):
    raw = env.unwrapped
    mujoco.mj_setState(raw.model, raw.data, payload["values"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(raw.model, raw.data)
    raw.np_random.bit_generator.state = payload["random_state"]
    env._elapsed_steps = payload["elapsed_steps"]


def save_training(agent, env, loop, checkpoint):
    payload = dict(loop=loop, environment=environment_state(env), readouts=agent.readouts)
    temporary = checkpoint.with_suffix(".tmp")
    agent.save(str(temporary), extra_state=payload)
    temporary.replace(checkpoint)


def train(agent, env, stage, checkpoint, resume_from=None):
    budget = BUDGETS[stage]
    modes = MUJOCO_MODE_PROFILES["Hopper-v4"]
    if resume_from is not None:
        payload = agent.load(str(resume_from))
        loop = payload["loop"]
        agent.readouts = payload["readouts"]
        apply_mode_to_env(env, "Hopper-v4", modes[loop["mode"]])
        env.reset(seed=1)
        restore_environment(env, payload["environment"])
    else:
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        apply_mode_to_env(env, "Hopper-v4", modes["normal"])
        state, _ = env.reset(seed=10000)
        agent.reset_context()
        loop = dict(step=0, episode=0, episode_age=0, reward_sum=0., state=state, mode="normal",
            mode_rng=np.random.RandomState(1), recent={k: deque(maxlen=agent.config.surprise_window)
                for k in ("state", "action", "reward", "context")}, rows=[], blocks=[], mode_events=[], returns=[])
    started, initial = time.monotonic(), loop["step"]
    for step in tqdm(range(initial, budget), initial=initial, total=budget, desc=agent.variant, mininterval=10):
        if step % 5000 == 0:
            mode = "normal" if step == 0 else str(loop["mode_rng"].choice(list(modes)))
            apply_mode_to_env(env, "Hopper-v4", modes[mode])
            loop["mode_events"].append(dict(step=step, previous_mode=loop["mode"], mode=mode))
            loop["mode"] = mode
        state = loop["state"]
        action = agent.select_action(state, deterministic=False)
        context = agent._current_context_np().copy()
        executed = np.clip(action * env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(executed)
        done = terminated or truncated
        if step == 0:
            loop["first_transition"] = dict(state=state.tolist(), action=action.tolist(), reward=float(reward),
                                           next_state=next_state.tolist(), done=bool(done))
        agent.readouts.observe(state, executed, next_state, loop["episode_age"])
        for k, v in (("state", state.copy()), ("action", action.copy()), ("reward", float(reward)), ("context", context)):
            loop["recent"][k].append(v)
        agent.store_transition(state, action, reward, next_state, float(done))
        metrics = agent.update(recent_rollout=_recent_rollout_payload(loop["recent"]))
        assert metrics is not None
        assert abs(metrics["effective_beta"] + 2. + agent.penalty_coefficient) < 1e-12
        metrics.update(signal=agent.readouts.signal, departure_coefficient=agent.readouts.departure.risk,
                       age_coefficient=agent.readouts.age_coefficient)
        loop["rows"].append({k: float(metrics[k]) for k in METRICS})
        loop["reward_sum"] += reward
        loop["episode_age"] += 1
        loop["state"], loop["step"] = next_state, step + 1
        if done:
            loop["returns"].append(float(loop["reward_sum"]))
            loop["episode"] += 1
            loop["episode_age"], loop["reward_sum"] = 0, 0.
            loop["state"], _ = env.reset(seed=10000 + loop["episode"])
            agent.reset_context()
        if (step + 1) % 1000 == 0 or step + 1 == budget:
            loop["blocks"].append(dict(step=step + 1, updates=len(loop["rows"]), mode=loop["mode"],
                **{k: float(np.mean([r[k] for r in loop["rows"]])) for k in METRICS}))
            loop["rows"].clear()
        if (step + 1) % 5000 == 0 or step + 1 == budget:
            save_training(agent, env, loop, checkpoint)
            print(f"[TRAIN] {agent.variant} steps={step+1}/{budget}, beta={metrics['effective_beta']:.5f}", flush=True)
    return dict(interactions=loop["step"], optimizer_updates=agent.training_steps - 99745,
        context_count=agent.context_tracker._count, replay_size=agent.replay_buffer.size,
        mode_events=loop["mode_events"], diagnostic_blocks=loop["blocks"], completed_episode_returns=loop["returns"],
        first_transition=loop["first_transition"], session_seconds=time.monotonic() - started, resumed_at=initial)


def evaluate(agent, stage):
    spec, snapshot, updates = stage_spec(stage), agent.snapshot_context(), agent.training_steps
    def rollout(env, count, seed):
        rewards, lengths, profiles = [], [], Counter()
        for episode in range(count):
            agent.restore_context(deepcopy(snapshot))
            result = _evaluate_agent(agent, env, 1000, 1, seed + episode)
            rewards.extend(result["rewards"])
            lengths.extend(result["lengths"])
            profiles.update(result["profile_counts"])
        env.close()
        return dict(mean=float(np.mean(rewards)), std=float(np.std(rewards)), rewards=rewards,
                    lengths=lengths, profile_counts=dict(profiles))
    def ood(schedule, profile=None):
        return ExtremeOODWrapper(gym.make("Hopper-v4"), config=ExtremeOODConfig(
            intensity=50., profile_family="structured", schedule=schedule, static_profile=profile,
            seed=spec["evaluation_start"], switch_every_steps=200, switch_probability=1.))
    agent.actor.eval()
    results = dict(id=rollout(gym.make("Hopper-v4"), spec["episodes"], spec["evaluation_start"]),
                   markov_50x=rollout(ood("markov"), spec["episodes"], spec["evaluation_start"]))
    probe = ood("static")
    profiles = [p["name"] for p in probe.profiles if p["name"] != "id"]
    probe.close()
    fixed = {p: rollout(ood("static", p), spec["fixed_episodes"], spec["evaluation_start"] + 100 * (i + 1))
             for i, p in enumerate(profiles)}
    agent.restore_context(snapshot)
    assert agent.training_steps == updates
    return results, fixed


def run_worker(args):
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    env = gym.make("Hopper-v4")
    agent = ContinuationAgent(env, args.variant)
    agent.start_shared(args.source_dir / SOURCE_NAME, shared)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{args.variant}_latest.pt"
    training = train(agent, env, args.stage, checkpoint, args.resume_from)
    env.close()
    # Evaluate a restored final checkpoint, not an unsaved in-memory actor.
    fresh = gym.make("Hopper-v4")
    restored = ContinuationAgent(fresh, args.variant)
    fresh.close()
    restored.load(str(checkpoint))
    results, fixed = evaluate(restored, args.stage)
    assert training["optimizer_updates"] == BUDGETS[args.stage]
    metadata = shared["metadata"]
    output = dict(revision=REVISION, stage=args.stage, variant=args.variant, continuation_seed=1,
        source_checkpoint=SOURCE_NAME, checkpoint_selection="final", shared_metadata=metadata, config=asdict(agent.config),
        training=training, added_training_budget=BUDGETS[args.stage], initial_replay_size=metadata["replay_size"],
        replay_refill_shared=True, reference_frozen=True, penalty_enters_actor_loss_only=True,
        source_training_state_available=False, resume_kind=metadata["resume_kind"],
        continuation_checkpoint_resumable=True, optimizer_updates_during_evaluation=0,
        restore_training_context_before_each_evaluation_episode=True,
        evaluation_seed_start=stage_spec(args.stage)["evaluation_start"], results=results, fixed_profile_results=fixed,
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
                     mujoco=mujoco.__version__, gpu=torch.cuda.get_device_name()))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_{args.variant}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {args.variant}: added={BUDGETS[args.stage]}, result={path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=BUDGETS, default="formal")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.variant:
        run_worker(args)
        return
    if not (args.shared_dir / "shared.pt").exists():
        prepare_shared(args.source_dir / SOURCE_NAME, args.shared_dir, args.stage)
    common = ["--stage", args.stage, "--source-dir", str(args.source_dir), "--shared-dir", str(args.shared_dir),
              "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    print(f"[BUNDLE] {args.stage}: {','.join(VARIANTS)}", flush=True)
    processes = {}
    for variant in VARIANTS:
        checkpoint = args.checkpoint_dir / f"{variant}_latest.pt"
        resume = ["--resume-from", str(checkpoint)] if checkpoint.exists() else []
        processes[variant] = subprocess.Popen([sys.executable, "-u", __file__, "--variant", variant, *common, *resume])
    codes = {v: p.wait() for v, p in processes.items()}
    print(f"[BUNDLE END] {codes}", flush=True)
    raise SystemExit(0 if all(c == 0 for c in codes.values()) else 1)


if __name__ == "__main__":
    main()
