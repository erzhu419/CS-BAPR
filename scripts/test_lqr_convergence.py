"""
CS-BAPR on LQR-v0: SINDy / NAU positive-case benchmark.

LQR is a positive case for the framework's architectural and symbolic claims:
  - Dynamics are linear: SINDy identifies A, B exactly (sparsity = 1).
  - Optimal policy is linear in state: u* = -K x.
    This is the unique setting in which NAU's L=0 derivative is also exact —
    a constant-derivative architecture matches a constant-Jacobian optimum.
  - OOD axis: initial-state amplitude m ∈ {1, 2, 5, 10, 20, 50}, the analog
    of the bus benchmark's `od_mult`. Training samples x0 in unit ball;
    OOD samples x0 in radius-m ball, requiring scale-aware extrapolation.

Usage:
    python scripts/test_lqr_convergence.py --episodes 500 --seed 0 --method csbapr
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from csbapr.config import CSBAPRConfig
from csbapr.agent import CSBAPRAgent
from csbapr.envs.lqr_env import LQREnv, EP_LEN


# ─────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────
def run_episode(env, agent, deterministic=False, train=False, train_freq=1):
    obs, _ = env.reset()
    total_reward = 0.0
    train_steps = 0
    for t in range(EP_LEN):
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

    return total_reward, train_steps


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate(env, agent, n_eval=5):
    rewards = []
    for _ in range(n_eval):
        r, _ = run_episode(env, agent, deterministic=True, train=False)
        rewards.append(r)
    return float(np.mean(rewards)), float(np.std(rewards))


def evaluate_ood(agent, amplitudes, n_eval=5, seed_base=10000):
    results = {}
    for m in amplitudes:
        rs = []
        for k in range(n_eval):
            env_ood = LQREnv(amplitude=m, seed=seed_base + k)
            r, _ = run_episode(env_ood, agent, deterministic=True, train=False)
            rs.append(r)
        results[m] = (float(np.mean(rs)), float(np.std(rs)))
        print(f"  amp_{m:4.1f}x: {np.mean(rs):10.2f} ± {np.std(rs):.2f}")
    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        type=str,
        default="csbapr",
        choices=["csbapr", "csbapr-no-sindy", "csbapr-no-nau", "bapr"],
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Config ──
    config = CSBAPRConfig()
    config.state_dim = 4
    config.action_dim = 2
    config.hidden_dim = 64
    config.num_critics = 3
    config.batch_size = 256
    config.warmup_steps = 1000
    config.lr_actor = 3e-4
    config.lr_critic = 3e-4

    if args.method == "csbapr":
        # Full CS-BAPR: NAU + SINDy + IRM + Jac
        config.use_nau_actor = True
        config.actor_type = None
        config.jac_weight = 0.01
        config.weight_sym = 0.01
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0
        config.sindy_lib_degree = 1  # linear basis suffices for LQR dynamics
        config.sindy_threshold = 0.001
        print("[CS-BAPR] NAU + SINDy + Jacobian Consistency (linear LQR)")
    elif args.method == "csbapr-no-sindy":
        config.use_nau_actor = True
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0
        print("[CS-BAPR-no-SINDy] NAU + training fixes (no symbolic alignment)")
    elif args.method == "csbapr-no-nau":
        config.use_nau_actor = False
        config.jac_weight = 0.01
        config.weight_sym = 0.01
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0
        config.sindy_lib_degree = 1
        config.sindy_threshold = 0.001
        print("[CS-BAPR-no-NAU] MLP + SINDy (architecture ablation)")
    else:  # bapr
        config.use_nau_actor = False
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 0.0
        print("[BAPR] Plain MLP baseline (no SINDy, no NAU, no fixes)")
        # Disable the new training fixes for true baseline
        # (effective_beta will still use the v10 P0+P1 by default — that's fine,
        #  it's part of the BAPR backbone we share.)

    agent = CSBAPRAgent(config.state_dim, config.action_dim, config)
    print(
        f"Config: state={config.state_dim}, action={config.action_dim}, "
        f"hidden={config.hidden_dim}, NAU={config.use_nau_actor}, "
        f"jac_w={config.jac_weight}, sym_w={config.weight_sym}"
    )

    env = LQREnv(amplitude=1.0, seed=args.seed)

    # ── SINDy Phase 0 (collect random trajectories, fit linear basis) ──
    # Action-aware SINDy toggle (the closed-loop-target fix flagged in §5.5
    # of the paper). Enables fitting dx/dt = f(x) + B u jointly so the
    # symbolic reference encodes both the open-loop dynamics A and the
    # input matrix B (still a state-only Jac-Consistency target until the
    # full closed-loop alignment is implemented).
    if os.environ.get("SINDY_WITH_CONTROL", "0") == "1":
        config.sindy_with_control = True
        print("[Config] sindy_with_control = True (action-aware Phase-0 fit)")
    use_sindy = (args.method in ("csbapr", "csbapr-no-nau"))
    if use_sindy:
        try:
            agent.sindy_preidentify(env)
            print(f"[Phase 0] SINDy identification done.")
        except Exception as e:
            print(f"[Phase 0] SINDy failed: {e!r}")
            print(f"[Phase 0] Falling back to no-SINDy mode")
            agent.f_sym_torch = None

    # ── Checkpoint paths (per method × seed) ──
    ckpt_dir = os.environ.get("CKPT_DIR", "/tmp/lqr_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"{args.method}_seed{args.seed}"
    last_ckpt = os.path.join(ckpt_dir, f"{tag}_last.pt")
    best_ckpt = os.path.join(ckpt_dir, f"{tag}_best.pt")
    meta_path = os.path.join(ckpt_dir, f"{tag}_meta.json")
    SAVE_EVERY = int(os.environ.get("CKPT_EVERY", "20"))

    # ── Resume from last checkpoint if present ──
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
            print(f"[RESUME] {tag}: starting at ep {start_ep}, best={best_total:.2f}")
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
    print(f"Training: {args.method} on LQR-v0, ep {start_ep}/{args.episodes}")
    print(f"{'=' * 60}")

    ep = start_ep - 1
    for ep in range(start_ep, args.episodes):
        ep_total, train_steps = run_episode(env, agent, deterministic=False, train=True)
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
                f"  ep {ep+1:4d}: total={ep_total:8.2f}, recent_25={recent:8.2f}, "
                f"best={best_total:8.2f}, t={elapsed:.0f}s"
            )

    # ── ID Evaluation ──
    print(f"\n{'=' * 60}")
    print("ID Evaluation (amplitude=1.0)")
    print(f"{'=' * 60}")
    id_mean, id_std = evaluate(env, agent, n_eval=20)
    print(f"  TOTAL : {id_mean:.2f} ± {id_std:.2f}")

    # ── OOD Evaluation: amplitude sweep ──
    print(f"\n{'=' * 60}")
    print("OOD Evaluation (amplitude sweep)")
    print(f"{'=' * 60}")
    evaluate_ood(agent, amplitudes=[1, 2, 5, 10, 20, 50], n_eval=20)

    total = time.time() - start
    print(f"\nTotal time: {total:.0f}s")


if __name__ == "__main__":
    main()
