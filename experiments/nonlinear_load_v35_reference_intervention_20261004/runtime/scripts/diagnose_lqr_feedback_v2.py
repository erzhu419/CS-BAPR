"""Read-only final-checkpoint diagnostics; no optimizer steps or model saves."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def alignment_metrics(actor, reference, observations):
    from csbapr.losses.jacobian import action_jacobian, deterministic_action

    device = next(actor.parameters()).device
    states = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device).requires_grad_(True)
    actions = deterministic_action(actor, states)
    targets = reference(states)
    jac_actor = action_jacobian(actions, states, create_graph=False)
    jac_ref = action_jacobian(targets, states, create_graph=False)
    with torch.no_grad():
        opposite = deterministic_action(actor, -states)
        result = {
            "n_states": len(states),
            "state_norm_median": states.norm(dim=-1).median().item(),
            "state_norm_p95": torch.quantile(states.norm(dim=-1), 0.95).item(),
            "unit_ball_state_fraction": (states.norm(dim=-1) <= 1).float().mean().item(),
            "action_component_rmse": (actions - targets).square().mean().sqrt().item(),
            "jacobian_squared_frobenius_mean": (jac_actor - jac_ref).square().sum((-2, -1)).mean().item(),
            "actor_jacobian_frobenius_mean": jac_actor.square().sum((-2, -1)).sqrt().mean().item(),
            "reference_jacobian_frobenius_mean": jac_ref.square().sum((-2, -1)).sqrt().mean().item(),
            "actor_saturated_component_fraction": (actions.abs() > 0.95).float().mean().item(),
            "reference_saturated_component_fraction": (targets.abs() > 0.95).float().mean().item(),
            "opposed_action_component_fraction": (actions * targets < 0).float().mean().item(),
            "oddness_component_rmse": (actions + opposite).square().mean().sqrt().item(),
        }
        if hasattr(actor, "nau_head"):
            features = actor.feature_net(states)
            mixing = actor.mix_alpha.sigmoid()
            linear = mixing * actor.nau_head(features)
            quadratic = (1 - mixing) * actor.nmu_head(actor.nmu_proj(features))
            result.update({
                "feature_saturated_fraction": (features.abs() > 0.95).float().mean().item(),
                "linear_mean_component_rms": linear.square().mean().sqrt().item(),
                "quadratic_mean_component_rms": quadratic.square().mean().sqrt().item(),
                "linear_mixing_weight": mixing.item(),
            })
    return result


def gradient_vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)]).detach()


def cosine(left, right):
    denominator = left.norm() * right.norm()
    return (torch.dot(left, right) / denominator).item() if denominator > 0 else None


def gradient_profile(agent, reference, n_batches=8):
    from csbapr.losses.jacobian import compute_jacobian_loss, deterministic_action
    from csbapr.losses.q_loss import compute_q_loss_csbapr
    from csbapr.utils import compute_reg_norm

    actor_parameters = list(agent.actor.parameters())
    critic_parameters = list(agent.critic.parameters())
    config = agent.config
    results = []
    for index in range(n_batches):
        np.random.seed(91000 + index)
        torch.manual_seed(91000 + index)
        sampled = agent.replay_buffer.sample(config.batch_size)
        state, action, reward, next_state, done = [torch.as_tensor(x, dtype=torch.float32) for x in sampled]
        reward, done = reward[:, None], done[:, None]
        variance = 0.99 * agent._reward_ema_var + 0.01 * reward.var().item()
        scaled_reward = reward / max(math.sqrt(variance), 1e-6)
        with torch.no_grad():
            next_action, next_log_prob, *_ = agent.actor.evaluate(next_state)
        reg_norm = compute_reg_norm(agent.target_critic)
        arguments = (state, action, scaled_reward, next_state, done,
                     next_action, next_log_prob, reg_norm)
        if agent.pi_sym_torch is not None:
            q_loss, predicted_q, ood_loss, weighted_lambda, symbolic_penalty = compute_q_loss_csbapr(
                agent.critic, agent.target_critic, agent.actor, agent.pi_sym_torch,
                *arguments, agent.belief_tracker, alpha=agent.alpha,
                gamma=config.gamma, weight_reg=config.weight_reg,
                weight_sym=config.weight_sym, beta_ood=config.beta_ood,
                penalty_decay_rate=0.1, device="cpu", batch_size=config.batch_size,
                enable_min_q_target=config.enable_min_q_target,
            )
        else:
            q_loss, predicted_q, ood_loss, weighted_lambda, symbolic_penalty = agent._compute_q_loss_bapr_fallback(
                *arguments, config.batch_size,
            )
        critic_gradient = gradient_vector(q_loss, critic_parameters)
        critic_terms = {"td_mse": q_loss - config.beta_ood * ood_loss,
                        "ensemble_std": config.beta_ood * ood_loss}
        critic_vectors = {name: gradient_vector(loss, critic_parameters)
                          for name, loss in critic_terms.items()}
        new_action, log_prob, *_ = agent.actor.evaluate(state)
        q_values = agent.critic(state, new_action)
        effective_lambda = 0 if agent.training_steps < config.bapr_warmup_iters else weighted_lambda
        effective_beta = -2 - effective_lambda * config.penalty_scale
        terms = {
            "value": -q_values.mean(),
            "risk": -effective_beta * q_values.std(dim=0).mean(),
            "entropy": agent.alpha * log_prob.mean(),
        }
        sac_loss = sum(terms.values())
        if agent.pi_sym_torch is not None and config.jac_weight > 0:
            terms["jacobian"] = config.jac_weight * compute_jacobian_loss(agent.actor, agent.pi_sym_torch, state)
        if hasattr(agent.actor, "regularization_loss") and config.nau_reg_weight > 0:
            terms["arithmetic_regularization"] = config.nau_reg_weight * agent.actor.regularization_loss()
        action_mse = (deterministic_action(agent.actor, state) - reference(state).detach()).square().mean()
        anchor_weight = getattr(config, "action_anchor_weight", 0.0)
        if anchor_weight > 0:
            terms["action_anchor"] = anchor_weight * action_mse
        vectors = {name: gradient_vector(loss, actor_parameters) for name, loss in terms.items()}
        sac_gradient = sum(vectors[name] for name in ("value", "risk", "entropy"))
        total_gradient = sum(vectors.values())
        anchor_gradient = gradient_vector(action_mse, actor_parameters)
        actor_norm = total_gradient.norm().item()
        critic_norm = critic_gradient.norm().item()
        results.append({
            "batch_seed": 91000 + index,
            "actor_terms": {name: {"weighted_loss": loss.item(), "gradient_l2": vectors[name].norm().item()}
                            for name, loss in terms.items()},
            "sac_loss": sac_loss.item(), "sac_gradient_l2": sac_gradient.norm().item(),
            "actor_total_gradient_l2": actor_norm,
            "actor_clip_multiplier": min(1, config.grad_clip / actor_norm) if config.grad_clip > 0 and actor_norm > 0 else 1,
            "sac_jacobian_gradient_cosine": cosine(sac_gradient, vectors["jacobian"]) if "jacobian" in vectors else None,
            "diagnostic_action_mse": action_mse.item(),
            "action_anchor_weight": anchor_weight,
            "diagnostic_action_mse_gradient_l2": anchor_gradient.norm().item(),
            "sac_action_mse_gradient_cosine": cosine(sac_gradient, anchor_gradient),
            "critic_total_loss": q_loss.item(), "critic_gradient_l2": critic_norm,
            "critic_terms": {name: {"weighted_loss": loss.item(), "gradient_l2": critic_vectors[name].norm().item()}
                             for name, loss in critic_terms.items()},
            "critic_clip_multiplier": min(1, config.grad_clip / critic_norm) if config.grad_clip > 0 and critic_norm > 0 else 1,
            "critic_q_mean": predicted_q.mean().item(), "critic_ensemble_std_mean": ood_loss.item(),
            "weighted_symbolic_target_shift": config.weight_sym * symbolic_penalty,
            "reward_normalization_std": math.sqrt(variance), "alpha": agent.alpha,
            "raw_reward_mean": reward.mean().item(), "normalized_reward_mean": scaled_reward.mean().item(),
            "critic_weight_regularization_target_shift_mean": (config.weight_reg * reg_norm).mean().item(),
            "effective_beta": effective_beta,
        })
    return results


def rollout_states(actor, amplitude):
    from csbapr.envs.lqr_env import EP_LEN, LQREnv
    from csbapr.losses.jacobian import deterministic_action

    envs = [LQREnv(amplitude=amplitude, seed=s) for s in range(10000, 10020)]
    states = np.stack([env.reset()[0] for env in envs])
    initial = states.copy()
    sampled, returns = [], np.zeros(len(envs))
    for step in range(EP_LEN):
        if step % 10 == 0:
            sampled.append(states.copy())
        with torch.no_grad():
            actions = deterministic_action(actor, torch.from_numpy(states)).numpy()
        transitions = [env.step(action) for env, action in zip(envs, actions)]
        states = np.stack([transition[0] for transition in transitions])
        returns += [transition[1] for transition in transitions]
        assert not any(t[2] or t[3] for t in transitions) or step == EP_LEN - 1
    return initial, np.concatenate(sampled), returns.tolist()


def diagnose(checkpoint, output):
    from csbapr.agent import CSBAPRAgent
    from csbapr.config import CSBAPRConfig
    from csbapr.envs.lqr_env import get_optimal_gain
    from csbapr.losses.jacobian import deterministic_action
    from csbapr.sindy.feedback import LQRFeedbackReference

    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    protocol = metadata["extra_state"]["protocol"]
    config = CSBAPRConfig(**protocol["config"])
    config.device = "cpu"
    del metadata
    agent = CSBAPRAgent(4, 2, config)
    state = agent.load(checkpoint)
    assert state["next_ep"] == 500 and state["env_steps"] == 100000 and agent.training_steps == 99001
    agent.actor.eval()
    agent.critic.eval()
    reference = agent.pi_sym_torch or LQRFeedbackReference(get_optimal_gain())
    reference.requires_grad_(False)
    origin = torch.zeros(1, 4)
    report = {
        "revision": f"{protocol['revision']}_readonly_diagnostic", "method": protocol["method"],
        "training_seed": protocol["training_seed"], "checkpoint_selection": "last",
        "training_updates": agent.training_steps, "optimizer_steps_taken": 0,
        "torch_version": torch.__version__, "analysis_device": "cpu",
        "reference": "saved_fitted_feedback" if agent.pi_sym_torch is not None else "known_model_feedback_diagnostic",
        "reference_gain_max_abs_error": float(np.max(np.abs(reference.gain.numpy() - get_optimal_gain()))),
        "origin_action": deterministic_action(agent.actor, origin).detach().tolist()[0],
        "origin": alignment_metrics(agent.actor, reference, np.zeros((1, 4))),
        "initial_states": {}, "on_policy_states": {}, "rollout_returns": {},
    }
    for amplitude in (1, 2, 5, 10, 20, 50):
        initial, observations, returns = rollout_states(agent.actor, amplitude)
        report["initial_states"][str(amplitude)] = alignment_metrics(agent.actor, reference, initial)
        report["on_policy_states"][str(amplitude)] = alignment_metrics(agent.actor, reference, observations)
        report["rollout_returns"][str(amplitude)] = returns
    np.random.seed(92000)
    replay_states = agent.replay_buffer.sample(2048)[0]
    report["replay_states"] = alignment_metrics(agent.actor, reference, replay_states)
    replay_norms = np.linalg.norm(np.stack([t[0] for t in agent.replay_buffer.buffer]), axis=1)
    report["replay_coverage"] = {
        "n_transitions": len(replay_norms),
        "unit_ball_state_fraction": float(np.mean(replay_norms <= 1)),
        "state_norm_median": float(np.median(replay_norms)),
        "state_norm_p95": float(np.quantile(replay_norms, 0.95)),
    }
    report["gradient_batches"] = gradient_profile(agent, reference)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Diagnostic complete: {report['method']} seed{report['training_seed']} -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    diagnose(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
