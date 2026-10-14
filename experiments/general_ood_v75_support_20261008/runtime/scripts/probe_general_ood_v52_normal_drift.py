"""Normal physics with frozen versus updating policy, no return-based tuning."""
import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from scripts.run_general_ood_v52 import SOURCE_NAME, METRICS
from scripts.train_csbapr import MUJOCO_MODE_PROFILES, apply_mode_to_env, _recent_rollout_payload
from scripts.v52_agent import ContinuationAgent

BUDGET = 1000


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    records = {}
    for arm in ("normal-frozen", "normal-updating"):
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2")
        agent.start_shared(args.source_dir / SOURCE_NAME, shared)
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        apply_mode_to_env(env, "Hopper-v4", MUJOCO_MODE_PROFILES["Hopper-v4"]["normal"])
        state, _ = env.reset(seed=10000)
        agent.reset_context()
        recent = {k: deque(maxlen=agent.config.surprise_window) for k in ("state", "action", "reward", "context")}
        before = [p.detach().clone() for p in agent.actor.parameters()]
        rows, episodes, age, first = [], 0, 0, None
        for step in tqdm(range(BUDGET), desc=arm, mininterval=10):
            action = agent.select_action(state, deterministic=False)
            context = agent._current_context_np().copy()
            executed = np.clip(action * env.action_space.high, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(executed)
            done = terminated or truncated
            if step == 0:
                first = dict(state=state.tolist(), action=action.tolist(), reward=float(reward),
                             next_state=next_state.tolist(), done=bool(done))
            agent.readouts.observe(state, executed, next_state, age)
            for k, v in (("state", state.copy()), ("action", action.copy()), ("reward", float(reward)), ("context", context)):
                recent[k].append(v)
            agent.store_transition(state, action, reward, next_state, float(done))
            metrics = agent.update(recent_rollout=_recent_rollout_payload(recent)) if arm == "normal-updating" else None
            rows.append(dict(signal=agent.readouts.signal, coefficient=agent.readouts.departure.risk,
                action_rms=float(np.sqrt(np.mean(action ** 2))), reward=float(reward), episode_age=age,
                q_loss=float(metrics["q_loss"]) if metrics else 0.))
            state, age = next_state, age + 1
            if done and step + 1 < BUDGET:
                episodes += 1
                age = 0
                state, _ = env.reset(seed=10000 + episodes)
                agent.reset_context()
        raw = env.unwrapped
        assert np.array_equal(raw.model.body_mass, env._original_body_mass)
        assert np.array_equal(raw.model.geom_friction, env._original_friction)
        assert np.array_equal(raw.model.opt.gravity, env._original_gravity)
        keys = ("signal", "coefficient", "action_rms", "reward", "q_loss")
        def means(window):
            return {k: float(np.mean([r[k] for r in window])) for k in keys}
        records[arm] = dict(first_transition=first, first16=means(rows[:16]), first50=means(rows[:50]),
            full1000=means(rows), first50_curves={k: [r[k] for r in rows[:50]] for k in ("signal", "coefficient")},
            bins50=[dict(start=i, end=i+50, **means(rows[i:i+50])) for i in range(0, BUDGET, 50)],
            updates=agent.training_steps-99745, physical_parameters_unchanged=True,
            actor_parameter_l2_change=float(torch.sqrt(sum((p.detach()-q).square().sum()
                for p,q in zip(agent.actor.parameters(),before))).item()))
        env.close()
        del agent
        torch.cuda.empty_cache()
    assert records["normal-frozen"]["first_transition"] == records["normal-updating"]["first_transition"]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = dict(revision="general_ood_v52_normal_drift", source_checkpoint=SOURCE_NAME,
        shared_reference_stage="preflight", environment_interactions=2000, optimizer_updates=1000,
        mode="normal-only", beta=-2., records=records)
    p = args.result_dir / "normal_drift.json"
    p.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] normal-only frozen/updating readout diagnosis: {p}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
