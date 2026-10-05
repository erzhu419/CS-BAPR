"""
LQR feedback-v2 entrypoint: controlled identification and action-space alignment.

The smooth arithmetic actor is compared against symbolic and head ablations.
The feedback reference uses fitted dynamics and normalized-action costs; its
tanh output is not the constrained LQR optimum. Results, evaluation-seed
returns, and checkpoints use a feedback_v2 prefix distinct from historical
manuscript evidence. Evaluation uses the final training checkpoint.

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
import math
import random
from dataclasses import asdict
from tqdm import tqdm

import numpy as np
import torch
import scipy
import pysindy
import sklearn

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from csbapr.config import CSBAPRConfig
from csbapr.agent import CSBAPRAgent
from csbapr.envs.lqr_env import LQREnv, EP_LEN, Q, R, ACTION_SCALE, get_optimal_gain
from csbapr.sindy.feedback import LQRFeedbackReference

SYMBOLIC_LQR_METHODS = (
    "csbapr", "csbapr-action-anchor", "csbapr-no-target-shift",
    "csbapr-action-anchor-no-target-shift",
    "csbapr-no-nau", "csbapr-no-nau-no-target-shift",
    "csbapr-centered-mlp", "csbapr-centered-mlp-no-target-shift",
    "csbapr-centered-mlp-critic-only-no-target-shift",
    "csbapr-centered-mlp-jacobian-only-no-target-shift",
)
LQR_METHODS = (
    *SYMBOLIC_LQR_METHODS,
    "csbapr-no-sindy", "csbapr-no-sindy-no-target-shift",
    "bapr", "bapr-no-target-shift", "resac", "resac-no-target-shift",
    "csbapr-no-nau-no-sindy", "csbapr-no-nau-no-sindy-no-target-shift",
    "csbapr-centered-mlp-no-sindy", "csbapr-centered-mlp-no-sindy-no-target-shift",
)


def make_env(environment, severity, seed):
    if environment == "initial_state":
        return LQREnv(amplitude=severity, seed=seed)
    if environment == "random_load":
        from csbapr.envs.random_load_env import RandomLoadEnv
        return RandomLoadEnv(severity=severity, seed=seed)
    if environment == "nonlinear_load":
        from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv
        return NonlinearLoadEnv(severity=severity, seed=seed)
    raise ValueError(f"Unknown LQR environment: {environment}")


def fitted_reference(model, environment):
    if environment in ("random_load", "nonlinear_load"):
        from csbapr.sindy.load_feedback import fitted_load_reference
        return fitted_load_reference(model)
    return LQRFeedbackReference.from_world_model(model, Q, R * ACTION_SCALE ** 2)


def known_reference(environment):
    if environment == "random_load":
        from csbapr.sindy.load_feedback import known_load_reference
        return known_load_reference()
    if environment == "nonlinear_load":
        from scripts.preflight_nonlinear_load_v9 import local_linear_gain
        return LQRFeedbackReference(local_linear_gain())
    return LQRFeedbackReference(get_optimal_gain())


# ─────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────
def run_episode(env, agent, deterministic=False, train=False, env_steps=0):
    obs, _ = env.reset()
    total_reward = 0.0
    train_steps = 0
    for t in range(EP_LEN):
        if train and env_steps < agent.config.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=deterministic)
        next_obs, reward, term, trunc, _ = env.step(action)

        if train:
            agent.replay_buffer.push(obs, action, reward, next_obs, float(term))
            env_steps += 1

        obs = next_obs
        total_reward += reward

        if train and env_steps >= agent.config.warmup_steps:
            if agent.replay_buffer.size >= agent.config.batch_size:
                agent._last_update_metrics = agent.update()
                train_steps += 1

        if term or trunc:
            break

    return total_reward, env_steps if train else train_steps


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate(env, agent, n_eval=5):
    rewards = []
    for _ in range(n_eval):
        r, _ = run_episode(env, agent, deterministic=True, train=False)
        rewards.append(r)
    return float(np.mean(rewards)), float(np.std(rewards))


def evaluate_ood(agent, amplitudes, n_eval=5, seed_base=10000, environment="initial_state"):
    results = {}
    for m in amplitudes:
        rs = []
        for k in range(n_eval):
            env_ood = make_env(environment, m, seed_base + k)
            r, _ = run_episode(env_ood, agent, deterministic=True, train=False)
            rs.append(r)
        results[m] = {"mean": float(np.mean(rs)), "std": float(np.std(rs)),
                      "returns": rs, "evaluation_seeds": list(range(seed_base, seed_base + n_eval))}
        label = "amp" if environment == "initial_state" else "load"
        print(f"  {label}_{m:4.1f}x: {np.mean(rs):10.2f} +/- {np.std(rs):.2f}")
    return results


def evaluate_feedback(reference, amplitudes, n_eval=20, seed_base=10000, environment="random_load"):
    """Evaluate the direct controller on the learner's paired disturbance paths."""
    device = reference.gain.device
    results = {}
    for severity in amplitudes:
        returns = []
        for seed in range(seed_base, seed_base + n_eval):
            env = make_env(environment, severity, seed)
            observation, _ = env.reset()
            total = 0.0
            for step in range(EP_LEN):
                with torch.no_grad():
                    inputs = torch.as_tensor(observation, dtype=torch.float32, device=device)
                    action = reference(inputs).cpu().numpy()
                observation, reward, terminated, truncated, _ = env.step(action)
                assert not truncated and terminated == (step == EP_LEN - 1)
                total += reward
            returns.append(total)
        results[severity] = {
            "mean": float(np.mean(returns)), "std": float(np.std(returns)),
            "returns": returns, "evaluation_seeds": list(range(seed_base, seed_base + n_eval)),
        }
    return {"gain": reference.gain.detach().cpu().tolist(), "amplitude_results": results}


def evaluate_inverse_feedback(amplitudes, n_eval=100, seed_base=50000):
    """Reuse the frozen pilot's feasible inverse controller on paired paths."""
    from scripts.preflight_nonlinear_load_v9 import rollout

    results = {}
    for severity in amplitudes:
        returns = [rollout("inverse_feedback", severity, seed, None)["return"]
                   for seed in range(seed_base, seed_base + n_eval)]
        results[severity] = {
            "mean": float(np.mean(returns)), "std": float(np.std(returns)),
            "returns": returns, "evaluation_seeds": list(range(seed_base, seed_base + n_eval)),
        }
    return {"physical_gain": get_optimal_gain().tolist(), "reference_is_optimal": False,
            "controller": "inverse_actuator_with_current_force_compensation",
            "amplitude_results": results}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def make_config(method, device="auto", environment="initial_state"):
    if method not in LQR_METHODS:
        raise ValueError(f"Unknown LQR method: {method}")
    base_method = method.removesuffix("-no-target-shift")
    config = CSBAPRConfig()
    config.env_name = "LQR-v0"
    config.state_dim = 4
    config.action_dim = 2
    config.hidden_dim = 64
    config.num_critics = 3
    config.batch_size = 256
    config.warmup_steps = 1000
    config.lr_actor = 3e-4
    config.lr_critic = 3e-4
    config.actor_type = "smooth-nau"
    config.track_L_eff = False  # Evaluate the exact certificate once, after training.
    config.device = device

    if base_method in ("csbapr", "csbapr-action-anchor"):
        # Full CS-BAPR: NAU + SINDy + IRM + Jac
        config.use_nau_actor = True
        config.actor_type = "smooth-nau"
        config.jac_weight = 0.01
        config.weight_sym = 0.01
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0
        config.sindy_lib_degree = 1  # linear basis suffices for LQR dynamics
        config.sindy_threshold = 0.001
        if base_method == "csbapr-action-anchor":
            config.action_anchor_weight = 1.0
        print("[CS-BAPR] NAU + SINDy + Jacobian Consistency (linear LQR)")
        if config.action_anchor_weight > 0:
            print("[Action anchor] Deterministic-action MSE, fixed weight 1.0")
    elif base_method == "csbapr-no-sindy":
        config.use_nau_actor = True
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0
        print("[CS-BAPR-no-SINDy] NAU + training fixes (no symbolic alignment)")
    elif base_method in ("csbapr-no-nau", "csbapr-no-nau-no-sindy",
                         "csbapr-centered-mlp", "csbapr-centered-mlp-no-sindy",
                         "csbapr-centered-mlp-critic-only", "csbapr-centered-mlp-jacobian-only"):
        config.use_nau_actor = False
        config.actor_type = "centered-tanh-mlp" if "centered" in base_method else "tanh-mlp"
        config.jac_weight = 0.0 if base_method.endswith("no-sindy") else 0.01
        config.weight_sym = config.jac_weight
        if base_method.endswith("critic-only"):
            config.jac_weight = 0.0
        elif base_method.endswith("jacobian-only"):
            config.weight_sym = 0.0
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0
        config.sindy_lib_degree = 1
        config.sindy_threshold = 0.001
        print(f"[No NAU] {config.actor_type}, jac_w={config.jac_weight}, sym_w={config.weight_sym}")
    elif base_method in ("bapr", "resac"):
        config.use_nau_actor = False
        config.actor_type = None
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 0.0
        if base_method == "resac":
            config.bapr_warmup_iters = 10**12
            config.penalty_scale = 0.0
            print("[RE-SAC] Shared ReLU baseline, fixed ensemble-risk penalty")
        else:
            print("[BAPR] ReLU baseline with inherited stabilization, without SINDy or NAU")
        # This baseline retains the shared stabilization defaults; it is not pristine BAPR.
    if method.endswith("-no-target-shift"):
        config.weight_reg = 0.0
        print("[Critic target] Positive weight-norm target shift removed")
    if environment in ("random_load", "nonlinear_load"):
        config.state_dim = 6
        config.env_name = "RandomLoad-v0" if environment == "random_load" else "NonlinearLoad-v0"
    elif environment != "initial_state":
        raise ValueError(f"Unknown LQR environment: {environment}")
    return config


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_training_checkpoint(agent, env, path, *, next_ep, env_steps, history, protocol):
    agent.save(path + ".tmp", extra_state={
        "next_ep": next_ep, "env_steps": env_steps, "history": history,
        "protocol": protocol, "env_rng": env._rng.bit_generator.state,
        "action_rng": env.action_space.np_random.bit_generator.state,
    })
    os.replace(path + ".tmp", path)


def restore_training_checkpoint(agent, env, path, protocol):
    state = agent.load(path)
    if state is None or state["protocol"] != protocol:
        raise ValueError("Resume requires a complete checkpoint from this exact protocol")
    env._rng.bit_generator.state = state["env_rng"]
    env.action_space.np_random.bit_generator.state = state["action_rng"]
    return state["next_ep"], state["env_steps"], state["history"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="csbapr", choices=LQR_METHODS)
    parser.add_argument("--revision", default="feedback_v2")
    parser.add_argument("--environment", choices=["initial_state", "random_load", "nonlinear_load"],
                        default="initial_state")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--evaluation-seed-start", type=int, default=10000)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "/tmp/lqr_feedback_v2_ckpt"))
    parser.add_argument("--result-dir", default=os.environ.get("RESULT_DIR", "results/lqr_feedback_v2"))
    parser.add_argument("--resume", help="Full-state checkpoint supplied by the scheduler")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_training(args.seed)
    config = make_config(args.method, args.device, args.environment)
    config.max_episodes = args.episodes

    agent = CSBAPRAgent(config.state_dim, config.action_dim, config)
    print(
        f"Config: state={config.state_dim}, action={config.action_dim}, "
        f"hidden={config.hidden_dim}, NAU={config.use_nau_actor}, "
        f"jac_w={config.jac_weight}, sym_w={config.weight_sym}"
    )

    env = make_env(args.environment, 1, args.seed)

    env.action_space.seed(args.seed)
    # ── Checkpoint paths (per method × seed) ──
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    tag = f"{args.revision}_{args.method}_seed{args.seed}"
    last_ckpt = os.path.join(args.ckpt_dir, f"{tag}_last.pt")
    SAVE_EVERY = int(os.environ.get("CKPT_EVERY", "20"))
    protocol = {"revision": args.revision, "method": args.method,
                "training_seed": args.seed, "training_episodes": args.episodes,
                "config": asdict(config)}
    if args.environment in ("random_load", "nonlinear_load"):
        protocol.update(environment=args.environment, environment_settings=env.specification())
    if args.evaluation_seed_start != 10000:
        protocol["evaluation_seed_start"] = args.evaluation_seed_start
    if args.n_eval != 20:
        protocol["evaluation_episodes_per_severity"] = args.n_eval

    # ── Resume from last checkpoint if present ──
    start_ep = 0
    env_steps = 0
    history = []
    resume_path = args.resume or (last_ckpt if os.path.exists(last_ckpt) else None)
    if resume_path:
        start_ep, env_steps, history = restore_training_checkpoint(agent, env, resume_path, protocol)
        print(f"[RESUME] {tag}: next episode {start_ep}, environment steps {env_steps}")
    else:
        if args.method in SYMBOLIC_LQR_METHODS:
            identify_env = make_env(args.environment, 1, args.seed)
            identify_env.action_space.seed(args.seed)
            agent.sindy_preidentify(
                identify_env, reference_builder=lambda model: fitted_reference(model, args.environment),
            )
        # Identification must not change training initial states or exploration RNGs.
        seed_training(args.seed)
        save_training_checkpoint(agent, env, last_ckpt, next_ep=0,
                                 env_steps=0, history=history, protocol=protocol)

    start = time.time()
    print(f"\n{'=' * 60}")
    print(f"Training: {args.method} on {config.env_name}, ep {start_ep}/{args.episodes}")
    print(f"{'=' * 60}")

    ep = start_ep - 1
    progress = tqdm(range(start_ep, args.episodes), initial=start_ep,
                    total=args.episodes, desc=f"{args.method}/seed{args.seed}",
                    unit="ep", mininterval=10)
    for ep in progress:
        ep_total, env_steps = run_episode(env, agent, train=True, env_steps=env_steps)
        history.append(ep_total)
        progress.set_postfix(return_=f"{ep_total:.2f}", updates=agent.training_steps, refresh=False)

        if (ep + 1) % SAVE_EVERY == 0 or (ep + 1) == args.episodes:
            save_training_checkpoint(agent, env, last_ckpt, next_ep=ep + 1,
                                     env_steps=env_steps, history=history, protocol=protocol)

        if (ep + 1) % 25 == 0:
            elapsed = time.time() - start
            recent = np.mean(history[-25:])
            print(
                f"  ep {ep+1:4d}: total={ep_total:8.2f}, recent_25={recent:8.2f}, "
                f"updates={agent.training_steps}, t={elapsed:.0f}s"
            )
            metrics = getattr(agent, "_last_update_metrics", None)
            if metrics is not None:
                print(f"    anchor_mse={metrics['action_anchor_loss']:.6f}, "
                      f"jacobian_error={metrics['jac_loss']:.6f}, alpha={metrics['alpha']:.6f}")

    # ── OOD Evaluation: amplitude sweep ──
    print(f"\n{'=' * 60}")
    print(f"OOD Evaluation ({args.environment} severity sweep)")
    print(f"{'=' * 60}")
    ood_results = evaluate_ood(agent, amplitudes=[1, 2, 5, 10, 20, 50], n_eval=args.n_eval,
                               seed_base=args.evaluation_seed_start, environment=args.environment)
    curvature = agent.actor.compute_L_eff()
    from scripts.diagnose_lqr_feedback_v2 import alignment_metrics
    reference = agent.pi_sym_torch or known_reference(args.environment).to(agent.device)
    policy_alignment = {
        "reference": "fitted_feedback" if agent.pi_sym_torch is not None else "known_model_diagnostic",
        "origin_action": agent.select_action(np.zeros(config.state_dim, dtype=np.float32), deterministic=True).tolist(),
        "initial_states": {
            str(amplitude): alignment_metrics(agent.actor, reference, np.stack([
                make_env(args.environment, amplitude, seed).reset()[0]
                for seed in range(args.evaluation_seed_start, args.evaluation_seed_start + args.n_eval)
            ])) for amplitude in (1, 2, 5, 10, 20, 50)
        },
    }
    controls = {}
    if args.environment in ("random_load", "nonlinear_load"):
        known_name = "known_local_linear" if args.environment == "nonlinear_load" else "known_model"
        references = {known_name: known_reference(args.environment).to(agent.device)}
        if agent.pi_sym_torch is not None:
            references["fitted_model"] = agent.pi_sym_torch
        for name, controller in references.items():
            controls[name] = evaluate_feedback(
                controller, [1, 2, 5, 10, 20, 50], seed_base=args.evaluation_seed_start,
                n_eval=args.n_eval, environment=args.environment,
            )
            print(f"Direct {name} feedback 50x mean return: {controls[name]['amplitude_results'][50]['mean']:.6f}")
        if args.environment == "nonlinear_load":
            controls["known_inverse"] = evaluate_inverse_feedback(
                [1, 2, 5, 10, 20, 50], n_eval=args.n_eval, seed_base=args.evaluation_seed_start,
            )
            print(f"Direct known_inverse feedback 50x mean return: {controls['known_inverse']['amplitude_results'][50]['mean']:.6f}")
    result_path = os.path.join(args.result_dir, f"{tag}_evaluation.json")
    with open(result_path + ".tmp", "w") as f:
        json.dump({**protocol, "actor_type": config.actor_type,
                   "checkpoint_selection": "last", "environment_steps": env_steps,
                   "training_updates": agent.training_steps, "training_returns": history,
                   "sindy": getattr(agent, "_sindy_report", None),
                   "runtime": {"python": sys.version, "torch": torch.__version__,
                               "numpy": np.__version__, "scipy": scipy.__version__,
                               "pysindy": pysindy.__version__, "sklearn": sklearn.__version__,
                               "device": str(agent.device),
                               "gpu": (torch.cuda.get_device_name()
                                       if torch.device(agent.device).type == "cuda" else None)},
                   "policy_curvature_bound": curvature if math.isfinite(curvature) else None,
                   "policy_alignment": policy_alignment,
                   "feedback_controls": controls,
                   "amplitude_results": ood_results}, f, indent=2, allow_nan=False)
    os.replace(result_path + ".tmp", result_path)

    total = time.time() - start
    print(f"\nTotal time: {total:.0f}s")
    print(f"Training complete: saved evaluation to {result_path}")


if __name__ == "__main__":
    main()
