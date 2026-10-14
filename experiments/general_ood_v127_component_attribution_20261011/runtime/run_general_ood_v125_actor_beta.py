"""Paired actor-beta continuation from every full 300k v123 source."""
import argparse
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

from csbapr.agent import CSBAPRAgent
import run_general_ood_v123_source_extension as v123
import analyze_general_ood_v124_source_stability as v124

v121, v122 = v123.v121, v123.v122
REVISION = "general_ood_v125_actor_beta"
ARMS = {"beta-minus2": -2., "beta-zero": 0.}


class ActorBetaAgent(CSBAPRAgent):
    def __init__(self, state_dim, action_dim, config, beta):
        super().__init__(state_dim, action_dim, config)
        self.actor_beta = beta

    def _effective_policy_beta(self, weighted_lambda):
        return self.actor_beta


def make_agent(env, device, beta):
    config = v121.v39.v38.make_config("Hopper-v4", "fixed-penalty", device)
    config.state_dim, config.action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    config.weight_reg = 0.
    agent = ActorBetaAgent(config.state_dim, config.action_dim, config, beta)
    agent.surprise_computer = v121.v39.RelativeSurpriseComputer(ema_alpha=config.surprise_ema)
    return agent


def stage_spec(arm, smoke=False):
    spec = v121.stage_spec(smoke)
    spec.update(steps=300000+(512 if smoke else 50000), source_steps=300000,
                study_revision=REVISION, intervention="actor-beta-only", arm=arm, effective_beta=ARMS[arm])
    return spec


def continue_training(agent, env, seed, spec, source_checkpoint, checkpoint, resume_from=None):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        saved = agent.load(str(source_checkpoint))
        assert (saved["revision"], saved["seed"], saved["spec"]) == (v121.REVISION, seed, v123.stage_spec())
        loop = saved["loop"]
        assert loop["step"] == spec["source_steps"] == agent.context_tracker._count
        assert agent.training_steps == loop["step"]-agent.config.batch_size+1 and agent.config.weight_reg == 0.
        env.reset(seed=0)
        v121.restore_environment(env, saved["environment"])
        if loop["episode_age"] == 0:
            loop["state"], _ = env.reset(seed=seed*10000+loop["episodes_completed"])
            agent.reset_context()
        loop["evaluations"].append(dict(step=loop["step"], **v121.evaluate(agent, spec["evaluation_seed"],
            spec["evaluation_episodes"], spec["evaluation_max_steps"])))
        v121.save_progress(agent, env, loop, spec, seed, checkpoint)
        resume_from = checkpoint
        print(f"[FORK] seed{seed} arm={spec['arm']} source_steps={loop['step']} "
              f"target_steps={spec['steps']} episode_age={loop['episode_age']} replay={agent.replay_buffer.size}", flush=True)
    return v121.train(agent, env, seed, spec, checkpoint, resume_from)


def evaluate_final(agent, smoke=False):
    spec = v124.stage_spec(smoke)
    results = {}
    for mode, noises in (("deterministic", (None,)), ("sampled", spec["noise_seeds"])):
        episodes = [v124.evaluate_episode(agent, reset, noise, spec["max_steps"])
                    for reset in spec["reset_seeds"] for noise in noises]
        results[mode] = dict(episodes=episodes, mean_length=float(np.mean([e["length"] for e in episodes])),
            mean_return=float(np.mean([e["episode_return"] for e in episodes])),
            mean_velocity=float(np.mean([e["mean_velocity"] for e in episodes])),
            falls=sum(e["physical_termination"] for e in episodes),
            time_limits=sum(e["time_limit_truncation"] for e in episodes))
    return dict(protocol=spec, device="cpu", modes=results, optimizer_updates=0,
                evaluation_interactions=sum(e["length"] for r in results.values() for e in r["episodes"]))


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    env = gym.make("Hopper-v4")
    agent = make_agent(env, args.device, ARMS[args.arm])
    source = json.loads((args.source_runtime/f"run_results/seed{args.seed}/source.json").read_text())
    assert (source["revision"], source["training_seed"], source["smoke"]) == (v123.REVISION, args.seed, False)
    assert asdict(agent.config) == dict(source["config"], device=args.device)
    runtime = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None)
    assert all(runtime[k] == source["runtime"][k] for k in runtime if k != "gpu")
    spec, started = stage_spec(args.arm, args.smoke), time.monotonic()
    checkpoint = args.source_runtime/f"run_checkpoints/seed{args.seed}/latest.pt"
    training = continue_training(agent, env, args.seed, spec, checkpoint, args.checkpoint_dir/"latest.pt", args.resume_from)
    final = args.checkpoint_dir/"final.pt"
    agent.save(str(final))
    indices = np.linspace(0, agent.replay_buffer.size-1, 32 if args.smoke else 512).astype(int)
    target_scale = v122.probe.decompose(agent, indices, 12200)
    restored = make_agent(env, "cpu", ARMS[args.arm])
    restored.load(str(final))
    env.close()
    final_evaluation = evaluate_final(restored, args.smoke)
    result = dict(revision=REVISION, training_seed=args.seed, arm=args.arm, effective_beta=ARMS[args.arm],
        smoke=args.smoke, spec=spec, config=asdict(agent.config), runtime=runtime,
        source_revision=v123.REVISION, source_checkpoint=str(checkpoint), source_interactions=300000,
        actual_new_training_interactions=spec["steps"]-spec["source_steps"],
        actual_new_optimizer_updates=training["optimizer_updates"]-source["training"]["optimizer_updates"],
        reward_settings=source["reward_settings"], training_mode="normal-only", trained_from_scratch=False,
        checkpoint=str(final), checkpoint_selection="final-not-best", periodic_evaluation_selects_checkpoint=False,
        source_readout_refitted=False, training=training, final_target_scale=target_scale,
        final_evaluation=final_evaluation, seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"actor_beta.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    length = final_evaluation["modes"]["deterministic"]["mean_length"]
    print(f"[DONE] seed{args.seed} arm={args.arm} new_steps={result['actual_new_training_interactions']} "
          f"beta={ARMS[args.arm]} deterministic_length={length:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
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
