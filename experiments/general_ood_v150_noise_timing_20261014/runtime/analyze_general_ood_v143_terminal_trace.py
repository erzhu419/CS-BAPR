"""Frozen CPU terminal-window traces and stable-ID cross-physics controls."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

import run_general_ood_v142_budget_extension as source_worker

REVISION = "general_ood_v143_terminal_trace"
CASES = [(seed, profile, profile) for seed in (3, 4) for profile in source_worker.PROFILES]
CASES += [(4, "id", p) for p in ("friction-only", "gravity-only")]
CASE_NAMES = tuple(f"seed{s}_{source}_to_{target}" for s, source, target in CASES)
v137, v138 = source_worker.v137, source_worker.source_worker.v138
evaluator = v137.v135


class TerminalTrace(gym.Wrapper):
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        raw = self.unwrapped
        self.steps, self.tail = 0, deque(maxlen=17)
        self.sum_abs_action = np.zeros(3)
        self.saturation_counts = np.zeros(3, dtype=np.int64)
        self.action_max = 0.
        self.bound = np.abs(raw.model.actuator_ctrlrange).max(axis=1)
        self.dofs = raw.model.jnt_dofadr[raw.model.actuator_trnid[:, 0]]
        self.trace_metadata = dict(joint_names=[raw.model.joint(i).name for i in range(raw.model.njnt)],
            actuator_joint_names=[raw.model.joint(int(i)).name for i in raw.model.actuator_trnid[:, 0]],
            actuator_ctrlrange=raw.model.actuator_ctrlrange.tolist(), actuator_gear=raw.model.actuator_gear.tolist(),
            healthy_height_bounds=[float(raw._healthy_z_range[0]), None],
            healthy_angle_bounds=list(raw._healthy_angle_range))
        return result

    def step(self, action):
        raw = self.unwrapped
        before = dict(qpos=raw.data.qpos.tolist(), qvel=raw.data.qvel.tolist())
        result = self.env.step(action)
        self.steps += 1
        effort = np.abs(np.asarray(action))/self.bound
        self.sum_abs_action += effort
        self.saturation_counts += effort >= .95
        self.action_max = max(self.action_max, float(effort.max()))
        if self.steps % 4 == 0 or result[2] or result[3]:
            self.tail.append(dict(step=self.steps, before=before, action=np.asarray(action).tolist(),
                after=dict(qpos=raw.data.qpos.tolist(), qvel=raw.data.qvel.tolist()),
                joint_torque=raw.data.qfrc_actuator[self.dofs].tolist(),
                physical_termination=bool(result[2]), time_limit_truncation=bool(result[3])))
        return result

    def summary(self):
        return dict(**self.trace_metadata, steps=self.steps, sampled_step_stride=4, max_snapshots=17,
            saturation_threshold=.95, mean_abs_normalized_action=(self.sum_abs_action/self.steps).tolist(),
            action_saturation_fraction=(self.saturation_counts/self.steps).tolist(),
            max_abs_normalized_action=self.action_max, tail=list(self.tail))


def evaluate_episode(agent, profile, cell, limit):
    recorder = []
    def factory(condition):
        assert condition == ("id" if profile == "id" else "static/"+profile)
        env = evaluator.make_env("id") if profile == "id" else v138.make_wrapped_env(condition)
        trace = TerminalTrace(env.env)
        env.env = trace
        recorder.append(trace)
        return env
    episode = evaluator.evaluate_episode(agent, cell, "online", limit, env_factory=factory)
    trace = recorder[0].summary()
    assert trace["steps"] == episode["length"] and trace["tail"][-1]["step"] == episode["length"]
    return dict(**episode, terminal_trace=trace)


def load_source(source):
    seed, profile = source["training_seed"], source["training_profile"]
    assert source["revision"] == source_worker.REVISION and not source["smoke"]
    assert source["spec"] == source_worker.stage_spec(profile)
    assert source["training"]["interactions"] == 600000 and source["training"]["optimizer_updates"] == 599745
    env, physics = source_worker.make_env(profile)
    try:
        assert source["physics"] == physics
        agent = source_worker.make_agent(env, "cpu", "mse-control")
        assert asdict(agent.config) == dict(source["config"], device="cpu")
        agent.load(source["checkpoint"])
    finally:
        env.close()
    assert agent.training_steps == 599745 and agent.context_tracker._count == 600000
    assert agent.replay_buffer.size == 0
    for model in (agent.actor, agent.critic, agent.target_critic):
        model.eval().requires_grad_(False)
    return agent


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seed, source_profile, profile = CASES[CASE_NAMES.index(args.case)]
    source = json.loads(args.source_json.read_text())
    assert (source["training_seed"], source["training_profile"]) == (seed, source_profile)
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
    assert all(versions[k] == source["runtime"][k] for k in versions)
    started = time.monotonic()
    agent = load_source(source)
    protocol = v137.evaluation_spec()
    spec = dict(protocol, case=args.case, source_profile=source_profile, evaluation_profile=profile,
        reset_seeds=[154000] if args.smoke else protocol["reset_seeds"],
        action_noise_seeds=[137000] if args.smoke else protocol["action_noise_seeds"],
        max_steps=256 if args.smoke else 1000, smoke=args.smoke)
    metadata = dict(revision=source["revision"], checkpoint=source["checkpoint"], seed=seed,
        profile=source_profile, config=source["config"], training_interactions=600000, optimizer_updates=599745)
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, runtime=versions, source=metadata, episodes=[])
    assert (progress["revision"], progress["spec"], progress["runtime"], progress["source"]) == (REVISION, spec, versions, metadata)
    condition = "id" if profile == "id" else "static/"+profile
    cells = [dict(condition=condition, mode="deterministic" if noise is None else "sampled", reset_seed=reset, noise_seed=noise)
             for reset in spec["reset_seeds"] for noise in (None, *spec["action_noise_seeds"])]
    assert [{k: e[k] for k in cells[0]} for e in progress["episodes"]] == cells[:len(progress["episodes"])]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells[len(progress["episodes"]):]:
        progress["episodes"].append(evaluate_episode(agent, profile, cell, spec["max_steps"]))
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        print(f"[EVAL] {args.case} {len(progress['episodes'])}/{len(cells)} length={progress['episodes'][-1]['length']}", flush=True)
    assert agent.training_steps == 599745 and agent.context_tracker._count == 600000 and agent.replay_buffer.size == 0
    assert not torch.cuda.is_initialized()
    row = dict(**progress, case=args.case, smoke=args.smoke, evaluation_device="cpu", cuda_initialized=False,
        new_training_interactions=0, optimizer_updates=0, replay_loaded=False,
        evaluation_interactions=sum(e["length"] for e in progress["episodes"]),
        intervention="Passive terminal-window instrumentation and frozen stable-ID cross-physics controls",
        reward_changes=[], healthy_termination_changes=[], physics_profile_changes=[], seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"terminal_trace.json").write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {args.case}: {len(cells)} episodes; zero updates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
