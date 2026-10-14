"""Normal behavior coverage with held-out learning nulls and paired changes."""
from collections import deque
from copy import deepcopy
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from csbapr.belief.departure import DepartureReadout, NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v43 as v43
from scripts.run_general_ood_v52 import SOURCE_NAME, environment_state, restore_environment
from scripts.train_csbapr import MUJOCO_MODE_PROFILES, apply_mode_to_env, _recent_rollout_payload
from scripts.v52_agent import ContinuationAgent, NormalDynamics

REVISION = "general_ood_v53_behavior_coverage"
MODES = tuple(MUJOCO_MODE_PROFILES["Hopper-v4"])
TRACE_KEYS = ("states", "actions", "executed_actions", "next_states", "rewards", "dones", "terminated", "truncated")
CHECKPOINT_STEPS = 500


def stage_spec(stage):
    if stage == "preflight":
        return dict(fit=[98200], calibration=[98210], test=[98220], steps=1200, switch_step=800)
    return dict(fit=[98300, 98301], calibration=[98310, 98311], test=list(range(98320, 98324)),
                steps=2000, switch_step=1000)


def trajectory_specs(spec):
    roles = (set(spec[k]) for k in ("fit", "calibration", "test"))
    fitting, calibration, testing = roles
    assert not (fitting & calibration or fitting & testing or calibration & testing)
    references = [dict(split=split, seed=seed, updating=updating, mode="normal")
        for split in ("fit", "calibration") for seed in spec[split] for updating in (False, True)]
    probes = [dict(split="test", seed=seed, updating=True, mode=mode)
              for seed in spec["test"] for mode in MODES]
    return references + probes


def save_progress(agent, env, progress, loop, path):
    temporary = path.with_suffix(".tmp")
    agent.save(str(temporary), extra_state=dict(progress=progress, loop=loop,
        environment=environment_state(env) if loop is not None else None))
    temporary.replace(path)


def collect(agent, env, cell, spec, loop=None, checkpoint=None):
    modes = MUJOCO_MODE_PROFILES["Hopper-v4"]
    if loop is None:
        random.seed(cell["seed"])
        np.random.seed(cell["seed"])
        torch.manual_seed(cell["seed"])
        apply_mode_to_env(env, "Hopper-v4", modes["normal"])
        state, _ = env.reset(seed=cell["seed"])
        agent.reset_context()
        loop = dict(step=0, state=state, episode=0, initial_updates=agent.training_steps,
            rows={k: [] for k in TRACE_KEYS}, q_losses=[],
            recent={k: deque(maxlen=agent.config.surprise_window) for k in ("state", "action", "reward", "context")})
    for step in tqdm(range(loop["step"], spec["steps"]), initial=loop["step"], total=spec["steps"],
                     desc=f"{cell['split']}/{cell['seed']}/{cell['updating']}/{cell['mode']}", mininterval=10):
        if cell["split"] == "test" and step == spec["switch_step"]:
            apply_mode_to_env(env, "Hopper-v4", modes[cell["mode"]])
        state = loop["state"]
        action = agent.select_action(state, deterministic=False)
        context = agent._current_context_np().copy()
        executed = np.clip(action * env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(executed)
        done = terminated or truncated
        for k, v in (("state", state.copy()), ("action", action.copy()), ("reward", float(reward)), ("context", context)):
            loop["recent"][k].append(v)
        agent.store_transition(state, action, reward, next_state, float(done))
        if cell["updating"]:
            metrics = agent.update(recent_rollout=_recent_rollout_payload(loop["recent"]))
            assert metrics is not None and metrics["effective_beta"] == -2.
            loop["q_losses"].append(float(metrics["q_loss"]))
        values = (state.copy(), action.copy(), executed.copy(), next_state.copy(), float(reward),
                  bool(done), bool(terminated), bool(truncated))
        for k, v in zip(TRACE_KEYS, values):
            loop["rows"][k].append(v)
        loop["state"], loop["step"] = next_state, step + 1
        if done and step + 1 < spec["steps"]:
            loop["episode"] += 1
            loop["state"], _ = env.reset(seed=cell["seed"] + 10000 * loop["episode"])
            agent.reset_context()
        if checkpoint is not None and (step + 1) % CHECKPOINT_STEPS == 0:
            checkpoint(loop)
    expected_mode = cell["mode"] if cell["split"] == "test" else "normal"
    raw = env.unwrapped
    np.testing.assert_array_equal(raw.model.body_mass, env._original_body_mass * modes[expected_mode]["body_mass"])
    np.testing.assert_array_equal(raw.model.geom_friction, env._original_friction * modes[expected_mode]["friction"])
    np.testing.assert_array_equal(raw.model.opt.gravity, env._original_gravity * modes[expected_mode]["gravity"])
    updates = agent.training_steps - loop["initial_updates"]
    assert updates == (spec["steps"] if cell["updating"] else 0)
    trace = {k: np.asarray(v) for k, v in loop["rows"].items()}
    record = dict(**cell, interactions=spec["steps"], optimizer_updates=updates,
        physical_terminations=int(trace["terminated"].sum()), time_limit_truncations=int(trace["truncated"].sum()),
        mean_q_loss=float(np.mean(loop["q_losses"])) if updates else None)
    return trace, record


def fit_reference(fitting, calibration):
    fitting, calibration = v43.concatenate_references(fitting), v43.concatenate_references(calibration)
    model = NormalDynamics(2, True).fit_references(fitting, calibration)
    scores, thresholds = model.calibrations["global-centered"].scores(model.errors(calibration), calibration["episode_steps"])
    reference = NormalAgeReference.fit_normal(np.log1p(scores / thresholds), calibration["episode_steps"])
    return model, reference


def observe(trace, model, reference, switch_step):
    ages = v43.episode_steps(trace["dones"])
    scores, thresholds = model.calibrations["global-centered"].scores(model.errors(trace), ages)
    signals = np.log1p(scores / thresholds)
    readout = DepartureReadout(reference)
    coefficients = np.asarray([readout.update(signal, age) for signal, age in zip(signals, ages)])
    windows = dict(full=(0, len(ages)), prefix=(0, switch_step), first_post50=(switch_step, switch_step + 50),
                   post=(switch_step, len(ages)), last64=(len(ages) - 64, len(ages)))
    return {name: dict(mean_coefficient=float(coefficients[start:end].mean()),
                      mean_signal=float(signals[start:end].mean())) for name, (start, end) in windows.items()}


def paired_prefix_equal(trace, control, switch_step):
    return (all(np.array_equal(trace[k][:switch_step], control[k][:switch_step]) for k in TRACE_KEYS)
            and all(np.array_equal(trace[k][switch_step], control[k][switch_step])
                    for k in ("states", "actions", "executed_actions")))


def summarize(records):
    tests = [r for r in records if r["split"] == "test"]
    normal = [r for r in tests if r["mode"] == "normal"]
    usable_null = all(r["observers"]["broad"][w]["mean_coefficient"] <= .1
                      for r in normal for w in ("full", "last64"))
    changes = [r for r in tests if r["mode"] != "normal"]
    separation = all(r["change_minus_nochange"]["broad"][w] > 0
                     for r in changes for w in ("first_post50", "last64"))
    return dict(normal_full_means={name: [r["observers"][name]["full"]["mean_coefficient"] for r in normal]
                                  for name in ("old", "broad")},
        normal_last64_means={name: [r["observers"][name]["last64"]["mean_coefficient"] for r in normal]
                            for name in ("old", "broad")},
        usable_learning_normal_null=usable_null, all_paired_first50_and_tail_positive=separation,
        decision="coverage-diagnostic-pass-not-RL-confirmation" if usable_null and separation else "hold-reference-coverage")


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec = stage_spec(args.stage)
    cells = trajectory_specs(spec)
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    source = args.source_dir / SOURCE_NAME
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{args.stage}_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(next_cell=0, fitting=[], calibration=[], models=None, records=[], control=None)
    if extra:
        assert extra["progress"]["stage"] == args.stage
    progress["stage"] = args.stage
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        if cell["split"] == "test" and progress["models"] is None:
            broad = fit_reference(progress["fitting"], progress["calibration"])
            progress["models"] = dict(old=(shared["model"], shared["reference"]), broad=broad)
            progress["fitting"], progress["calibration"] = [], []
            print("[REFERENCE] Frozen normal + learning-normal fit and calibration complete", flush=True)
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2")
        agent.start_shared(source, shared)
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            mode = cell["mode"] if cell["split"] == "test" and loop["step"] > spec["switch_step"] else "normal"
            apply_mode_to_env(env, "Hopper-v4", MUJOCO_MODE_PROFILES["Hopper-v4"][mode])
            env.reset(seed=cell["seed"])
            restore_environment(env, extra["environment"])
        trace, record = collect(agent, env, cell, spec, loop,
            lambda active: save_progress(agent, env, progress, active, checkpoint))
        if cell["split"] != "test":
            progress["fitting" if cell["split"] == "fit" else "calibration"].append(trace)
            record["reference_record"] = v43.reference_record(trace, cell["seed"], cell["split"])
        else:
            record["observers"] = {name: observe(trace, *model, spec["switch_step"])
                                    for name, model in progress["models"].items()}
            if cell["mode"] == "normal":
                progress["control"] = (trace, deepcopy(record))
            control, baseline = progress["control"]
            assert baseline["seed"] == cell["seed"]
            record["paired_prefix_equal"] = paired_prefix_equal(trace, control, spec["switch_step"])
            assert record["paired_prefix_equal"], "mismatched-learning-prefix"
            record["change_minus_nochange"] = {name: {w: values["mean_coefficient"] -
                baseline["observers"][name][w]["mean_coefficient"] for w, values in windows.items()}
                for name, windows in record["observers"].items()}
        progress["records"].append(record)
        progress["next_cell"] = index + 1
        save_progress(agent, env, progress, None, checkpoint)
        print(f"[CELL] {index+1}/{len(cells)}: {cell}, updates={record['optimizer_updates']}", flush=True)
        extra = None
        env.close()
        del agent
        torch.cuda.empty_cache()
    records = progress["records"]
    output = dict(revision=REVISION, stage=args.stage, stage_spec=spec, source_checkpoint=SOURCE_NAME,
        source_training_seed=1, source_updates=99745, shared_replay_size=shared["replay"].size,
        shared_replay_origin="v52 frozen-policy normal fit-only; never augmented with held-out reference/test trajectories",
        action_sampling="stochastic", beta=-2., auxiliary_controls_actions=False, episode_time_limit=1000,
        readout_settings=SETTINGS, reference_frozen_during_test=True,
        environment_interactions=sum(r["interactions"] for r in records),
        optimizer_updates=sum(r["optimizer_updates"] for r in records),
        reference_metadata={name: dict(model=model.metadata(), age_reference=reference.metadata())
                            for name, (model, reference) in progress["models"].items()},
        records=records, summary=summarize(records),
        limitations=["One source training checkpoint; continuation seeds are not independent source training seeds.",
                     "Readouts are measured under fixed beta and do not establish adaptive-penalty return benefits.",
                     "Physical profile changes are diagnostic perturbations, not a 50x OOD return comparison."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / "behavior_coverage.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {output['summary']['decision']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), required=True)
    for name in ("source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
