"""Decompose frozen v121 Bellman targets without optimizer or environment steps."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

import run_general_ood_v121_locomotion_source as v121
from csbapr.utils import compute_reg_norm

REVISION = "general_ood_v122_target_scale"


@torch.no_grad()
def decompose(agent, indices, noise_seed):
    sample = [agent.replay_buffer.buffer[int(i)] for i in indices]
    arrays = [np.stack([r[k] for r in sample]) for k in range(7)]
    states, actions, rewards, next_states, dones, contexts, next_contexts = [
        torch.as_tensor(x, dtype=torch.float32, device=agent.device) for x in arrays]
    states = agent._augment_state_tensor(states, contexts)
    next_states = agent._augment_state_tensor(next_states, next_contexts)
    with torch.random.fork_rng():
        torch.manual_seed(noise_seed)
        next_actions, next_log_prob, *_ = agent.actor.evaluate(next_states)
    next_var = .99*agent._reward_ema_var+.01*float(rewards.var().item())
    reward_std = max(next_var**.5, 1e-6) if agent.config.enable_reward_ema else 1.
    normalized = rewards/reward_std
    bootstrap = agent.config.gamma*(1.-dones)
    minimum = agent.target_critic(next_states, next_actions).min(dim=0)[0]
    entropy = -agent.alpha*next_log_prob
    norms = compute_reg_norm(agent.target_critic)
    uplift = agent.config.weight_reg*norms[:, None]
    target_without = normalized[None, :]+bootstrap[None, :]*(minimum[None, :]+entropy[None, :])
    target = target_without+bootstrap[None, :]*uplift
    predicted = agent.critic(states, actions)
    expected_loss = torch.nn.functional.mse_loss(predicted, target)+agent.config.beta_ood*predicted.std(0).mean()
    actual_loss, *_ = agent._compute_q_loss_bapr_fallback(states, actions, normalized[:, None], next_states,
        dones[:, None], next_actions, next_log_prob, norms, len(indices))
    torch.testing.assert_close(actual_loss, expected_loss)
    discounted_uplift = float((bootstrap[None, :]*uplift).mean().item())
    mean_reward = float(normalized.mean().item())
    return dict(samples=len(indices), noise_seed=noise_seed, reward_normalization_std=reward_std,
        mean_raw_reward=float(rewards.mean().item()), mean_normalized_reward=mean_reward,
        mean_discounted_min_next_q=float((bootstrap*minimum).mean().item()),
        mean_discounted_entropy_term=float((bootstrap*entropy).mean().item()),
        target_critic_l1_per_critic=norms.tolist(), target_l1_uplift_per_critic=uplift[:, 0].tolist(),
        mean_discounted_l1_uplift=discounted_uplift,
        l1_uplift_over_abs_normalized_reward=discounted_uplift/abs(mean_reward) if mean_reward else None,
        mean_target=float(target.mean().item()), mean_target_without_l1=float(target_without.mean().item()),
        mean_predicted_q=float(predicted.mean().item()), alpha=float(agent.alpha),
        mean_next_log_prob=float(next_log_prob.mean().item()), done_fraction=float(dones.mean().item()),
        actual_loss_reconstructed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = []
    for seed in v121.SEEDS:
        env = gym.make("Hopper-v4")
        agent = v121.v39.make_agent("Hopper-v4", "fixed-minus2", "cpu", env)
        env.close()
        checkpoint = args.source_runtime/f"run_checkpoints/seed{seed}/latest.pt"
        extra = agent.load(str(checkpoint))
        assert extra["revision"] == v121.REVISION and extra["loop"]["step"] == 100000
        assert agent.training_steps == 99745 and agent.replay_buffer.size == 100000
        rows = {}
        for pool, start in (("uniform_training", 0), ("last10k", 90000)):
            indices = np.linspace(start, 99999, 512).astype(int)
            rows[pool] = [decompose(agent, indices, noise) for noise in (12200, 12201, 12202)]
        records.append(dict(training_seed=seed, checkpoint=str(checkpoint), pools=rows))
        del agent, extra
    output = dict(revision=REVISION, records=records, optimizer_updates=0, environment_step_calls=0,
        scope="frozen endpoint decomposition; removing a term here does not establish training causality")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"target_scale.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print("[DONE] v121 target decomposition: three sources, zero optimizer/environment steps", flush=True)


if __name__ == "__main__":
    main()
