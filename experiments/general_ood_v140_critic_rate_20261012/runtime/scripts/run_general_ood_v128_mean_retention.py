"""Matched mean-only source retention from full 300k Hopper snapshots."""
import argparse
from copy import deepcopy
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

import run_general_ood_v125_actor_beta as v125
import analyze_general_ood_v126_noise_sensitivity as v126

v121, v123 = v125.v121, v125.v123
REVISION = "general_ood_v128_mean_retention"
ARMS = {"unchanged-control": 0., "mean-only-source-regularization": 1.}
METRICS = ("mean_shift_kl", "mean_retention_loss", "reference_action_mse")
NOISE_GROUPS = {"legacy": [12400, 12401, 12402], "v126_seen": [12600, 12601, 12602],
                "new_heldout": [12800, 12801, 12802]}


class MeanRetentionAgent(v125.ActorBetaAgent):
    def __init__(self, state_dim, action_dim, config, coefficient):
        super().__init__(state_dim, action_dim, config, -2.)
        self.retention_coefficient = coefficient
        self.reference_actor = None
        self.retention_metrics = {}

    def set_reference(self, actor_state):
        self.reference_actor = deepcopy(self.actor)
        self.reference_actor.load_state_dict(actor_state)
        self.reference_actor.eval().requires_grad_(False)

    def _actor_auxiliary_loss(self, states):
        assert self.reference_actor is not None
        mean, _ = self.actor(states)
        with torch.no_grad():
            reference_mean, reference_log_std = self.reference_actor(states)
        divergence = .5 * ((mean-reference_mean)/reference_log_std.exp()).square().sum(-1).mean()
        alpha = torch.as_tensor(self.alpha, device=states.device, dtype=states.dtype).detach()
        penalty = self.retention_coefficient * alpha * divergence
        self.retention_metrics = dict(mean_shift_kl=float(divergence.detach()),
            mean_retention_loss=float(penalty.detach()),
            reference_action_mse=float((mean.tanh()-reference_mean.tanh()).square().mean().detach()))
        return penalty if self.retention_coefficient else None

    def update(self, *args, **kwargs):
        metrics = super().update(*args, **kwargs)
        if metrics is not None:
            metrics.update(self.retention_metrics)
        return metrics


def make_agent(env, device, arm):
    config = v121.v39.v38.make_config("Hopper-v4", "fixed-penalty", device)
    config.state_dim, config.action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    config.weight_reg = 0.
    agent = MeanRetentionAgent(config.state_dim, config.action_dim, config, ARMS[arm])
    agent.surprise_computer = v121.v39.RelativeSurpriseComputer(ema_alpha=config.surprise_ema)
    return agent


def stage_spec(arm, smoke=False):
    spec = v125.stage_spec("beta-minus2", smoke)
    spec.update(study_revision=REVISION, intervention="mean-only-source-regularization", arm=arm,
        mean_retention_coefficient=ARMS[arm], reference_steps=300000,
        retention_input="same current augmented replay states", retention_alpha="detached current alpha",
        retention_distance="0.5 * mean_batch(sum_action(((mu-mu_ref)/std_ref)^2))")
    return spec


def evaluation_spec(smoke=False):
    return dict(device="cpu", reset_seeds=list(range(150000, 150001 if smoke else 150005)),
        noise_groups={k: v[:1] if smoke else v for k, v in NOISE_GROUPS.items()},
        max_steps=1000, sampled_noise_scale=1., noise_schedule="v126 CPU step-sized float32 Gaussian draws")


def evaluate_final(agent, smoke=False):
    spec = evaluation_spec(smoke)
    groups = {"deterministic": [None], **spec["noise_groups"]}
    result = {}
    for group, noises in groups.items():
        episodes = [v126.evaluate_episode(agent, reset, 0. if noise is None else 1., noise, spec["max_steps"])
                    for reset in spec["reset_seeds"] for noise in noises]
        result[group] = dict(episodes=episodes, mean_length=float(np.mean([e["length"] for e in episodes])),
            mean_return=float(np.mean([e["episode_return"] for e in episodes])),
            mean_velocity=float(np.mean([e["mean_velocity"] for e in episodes])),
            falls=sum(e["physical_termination"] for e in episodes),
            time_limits=sum(e["time_limit_truncation"] for e in episodes))
    return dict(protocol=spec, device="cpu", groups=result, optimizer_updates=0,
        evaluation_interactions=sum(e["length"] for r in result.values() for e in r["episodes"]))


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    v121.METRICS = tuple(dict.fromkeys((*v121.METRICS, *METRICS)))
    env = gym.make("Hopper-v4")
    agent = make_agent(env, args.device, args.arm)
    source = json.loads((args.source_runtime/f"run_results/seed{args.seed}/source.json").read_text())
    assert (source["revision"], source["training_seed"], source["smoke"]) == (v123.REVISION, args.seed, False)
    assert source["training"]["interactions"] == 300000 and source["checkpoint_selection"] == "final-not-best"
    assert asdict(agent.config) == dict(source["config"], device=args.device)
    runtime = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None)
    assert all(runtime[k] == source["runtime"][k] for k in runtime if k != "gpu")
    reference_checkpoint = source["checkpoint"]
    reference = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
    agent.set_reference(reference["actor"])
    del reference
    spec, started = stage_spec(args.arm, args.smoke), time.monotonic()
    checkpoint = args.source_runtime/f"run_checkpoints/seed{args.seed}/latest.pt"
    training = v125.continue_training(agent, env, args.seed, spec, checkpoint,
        args.checkpoint_dir/"latest.pt", args.resume_from)
    final = args.checkpoint_dir/"final.pt"
    agent.save(str(final))
    indices = np.linspace(0, agent.replay_buffer.size-1, 32 if args.smoke else 512).astype(int)
    target_scale = v125.v122.probe.decompose(agent, indices, 12200)
    restored = v125.make_agent(env, "cpu", -2.)
    restored.load(str(final))
    env.close()
    final_evaluation = evaluate_final(restored, args.smoke)
    result = dict(revision=REVISION, training_seed=args.seed, arm=args.arm,
        mean_retention_coefficient=ARMS[args.arm], effective_beta=-2., smoke=args.smoke,
        spec=spec, config=asdict(agent.config), runtime=runtime, source_revision=v123.REVISION,
        source_checkpoint=str(checkpoint), reference_checkpoint=reference_checkpoint, source_interactions=300000,
        actual_new_training_interactions=spec["steps"]-spec["source_steps"],
        actual_new_optimizer_updates=training["optimizer_updates"]-source["training"]["optimizer_updates"],
        reward_settings=source["reward_settings"], reward_or_physics_changes=[], training_mode="normal-only",
        trained_from_scratch=False, checkpoint=str(final), checkpoint_selection="final-not-best",
        periodic_evaluation_selects_checkpoint=False, source_readout_refitted=False,
        reference_frozen=True, current_std_directly_regularized=False, initial_normalizer_swapped=False,
        training=training, final_target_scale=target_scale, final_evaluation=final_evaluation,
        seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"mean_retention.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} arm={args.arm} new_steps={result['actual_new_training_interactions']} "
          f"det_length={final_evaluation['groups']['deterministic']['mean_length']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
