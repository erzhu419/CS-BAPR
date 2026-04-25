"""
CS-BAPR on Hopper-v4 with wind-disturbance OOD.

Train with no wind. Evaluate at amplitudes {0, 1, 2, 5, 10, 20} (units of
body-mass × g) in the +x direction. amp=0 reproduces the unperturbed env;
larger amplitudes form a continuous, gentle OOD axis analogous to the
bus benchmark's od_mult.

Methods: csbapr (NAU), csbapr-no-nau (MLP+fixes), csbapr-relu (ReLU),
         bapr (plain ReLU baseline).

SINDy/IRM are disabled on Hopper — its dynamics include unilateral contact,
which is outside the smooth-ODE regime SINDy is designed for. The point of
this benchmark is to validate that the architecture+training-fix story
generalizes from transit holding to MuJoCo locomotion.

Usage:
    python scripts/test_hopper_wind.py --episodes 500 --seed 0 --method csbapr
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import gymnasium as gym
from csbapr.config import CSBAPRConfig
from csbapr.agent import CSBAPRAgent
from csbapr.envs.wind_wrapper import WindDisturbanceWrapper


def make_env(amplitude: float = 0.0, seed: int | None = None):
    env = gym.make("Hopper-v4")
    env = WindDisturbanceWrapper(env, body_name="torso", direction=(1.0, 0.0, 0.0))
    env.set_wind_amplitude(amplitude)
    if seed is not None:
        env.reset(seed=seed)
    return env


def run_episode(env, agent, deterministic=False, train=False, train_freq=1, max_steps=1000):
    obs, _ = env.reset()
    total_reward = 0.0
    train_steps = 0
    for t in range(max_steps):
        action = agent.select_action(obs, deterministic=deterministic)
        next_obs, reward, term, trunc, _ = env.step(action)

        if train:
            agent.replay_buffer.push(obs, action, reward, next_obs, float(term))

        obs = next_obs
        total_reward += reward

        if train and train_freq > 0 and (t + 1) % train_freq == 0:
            if agent.replay_buffer.size >= agent.config.batch_size:
                agent.update()
                train_steps += 1

        if term or trunc:
            break

    return total_reward, t + 1, train_steps


def evaluate_at_wind(agent, amplitude, n_eval=5, seed_base=10000):
    rewards = []
    lengths = []
    for k in range(n_eval):
        env = make_env(amplitude=amplitude, seed=seed_base + k)
        r, T, _ = run_episode(env, agent, deterministic=True, train=False)
        rewards.append(r)
        lengths.append(T)
    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(lengths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        type=str,
        default="csbapr",
        choices=["csbapr", "csbapr-no-nau", "csbapr-relu", "bapr"],
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dim = 11  # Hopper observation
    action_dim = 3  # Hopper action

    config = CSBAPRConfig()
    config.state_dim = state_dim
    config.action_dim = action_dim
    config.hidden_dim = 128
    config.num_critics = 3
    config.batch_size = 256
    config.warmup_steps = 1000
    config.lr_actor = 3e-4
    config.lr_critic = 3e-4
    # SINDy / IRM off on Hopper (contact dynamics violate SINDy assumptions)
    config.jac_weight = 0.0
    config.weight_sym = 0.0
    config.beta_bc = 0.0

    if args.method == "csbapr":
        config.use_nau_actor = True
        config.actor_type = None
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        print("[CS-BAPR] NAU actor + training fixes (no SINDy on Hopper)")
    elif args.method == "csbapr-no-nau":
        config.use_nau_actor = False
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 1e-4
        print("[CS-BAPR-no-NAU] MLP + training fixes (architecture ablation)")
    elif args.method == "csbapr-relu":
        # NAU's structural alternative — still gets the training fixes.
        config.use_nau_actor = False
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 0.0
        print("[CS-BAPR-ReLU] MLP + training fixes (= csbapr-no-nau without weight decay)")
    else:  # bapr
        config.use_nau_actor = False
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 0.0
        # Disable the four training-stabilization fixes too: this is the true
        # pre-fix baseline. The agent's effective_beta still uses the v10
        # warmup/penalty_scale defaults, but nothing else.
        print("[BAPR] Plain MLP baseline")

    agent = CSBAPRAgent(state_dim, action_dim, config)
    print(
        f"Config: hidden={config.hidden_dim}, NAU={config.use_nau_actor}, "
        f"warmup={config.bapr_warmup_iters}, scale={config.penalty_scale}"
    )

    env = make_env(amplitude=0.0, seed=args.seed)

    # ── Checkpoint paths ──
    ckpt_dir = os.environ.get("CKPT_DIR", "/tmp/hopper_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"{args.method}_seed{args.seed}"
    last_ckpt = os.path.join(ckpt_dir, f"{tag}_last.pt")
    best_ckpt = os.path.join(ckpt_dir, f"{tag}_best.pt")
    meta_path = os.path.join(ckpt_dir, f"{tag}_meta.json")
    SAVE_EVERY = int(os.environ.get("CKPT_EVERY", "20"))

    # Resume
    start_ep = 0
    best_total = -float("inf")
    history = []
    if os.path.exists(last_ckpt) and os.path.exists(meta_path):
        try:
            agent.load(last_ckpt)
            with open(meta_path) as f:
                meta = json.load(f)
            start_ep = int(meta.get("next_ep", 0))
            best_total = float(meta.get("best_total", -float("inf")))
            history = list(meta.get("history", []))
            print(f"[RESUME] {tag}: starting at ep {start_ep}, best={best_total:.1f}")
        except Exception as e:
            print(f"[RESUME] failed: {e!r}")

    def _atomic_save_meta(ep_just_done):
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "next_ep": ep_just_done + 1,
                    "best_total": best_total,
                    "history": history,
                    "method": args.method,
                    "seed": args.seed,
                },
                f,
            )
        os.replace(tmp, meta_path)

    start = time.time()
    print(f"\n{'=' * 60}")
    print(f"Training: {args.method} on Hopper-v4 (no wind), ep {start_ep}/{args.episodes}")
    print(f"{'=' * 60}")

    ep = start_ep - 1
    for ep in range(start_ep, args.episodes):
        ep_total, ep_len, _ = run_episode(
            env, agent, deterministic=False, train=True, max_steps=args.max_steps
        )
        history.append(ep_total)

        if ep_total > best_total:
            best_total = ep_total
            agent.save(best_ckpt + ".tmp")
            os.replace(best_ckpt + ".tmp", best_ckpt)

        if (ep + 1) % SAVE_EVERY == 0 or (ep + 1) == args.episodes:
            agent.save(last_ckpt + ".tmp")
            os.replace(last_ckpt + ".tmp", last_ckpt)
            _atomic_save_meta(ep)

        if (ep + 1) % 25 == 0:
            elapsed = time.time() - start
            recent = np.mean(history[-25:])
            print(
                f"  ep {ep+1:4d}: total={ep_total:7.1f}, recent_25={recent:7.1f}, "
                f"len={ep_len}, best={best_total:7.1f}, t={elapsed:.0f}s"
            )

    # ── ID Evaluation ──
    print(f"\n{'=' * 60}")
    print("ID Evaluation (wind=0)")
    print(f"{'=' * 60}")
    id_mean, id_std, id_len = evaluate_at_wind(agent, 0.0, n_eval=10)
    print(f"  TOTAL : {id_mean:.1f} ± {id_std:.1f} (mean ep_len={id_len:.0f})")

    # ── OOD Wind Sweep ──
    print(f"\n{'=' * 60}")
    print("OOD Evaluation (wind amplitude sweep)")
    print(f"{'=' * 60}")
    for amp in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]:
        m, s, L = evaluate_at_wind(agent, amp, n_eval=10)
        print(f"  wind_{amp:5.1f}x: {m:7.1f} ± {s:.1f}  (mean_len={L:.0f})")

    total = time.time() - start
    print(f"\nTotal time: {total:.0f}s")


if __name__ == "__main__":
    main()
