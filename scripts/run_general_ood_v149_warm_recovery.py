"""Matched full-state SAC continuation from stable v147 no-pulse sources."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[0]/"experiments/general_ood_v149_warm_recovery_20261014/runtime"
           if HERE.name == "scripts" else HERE)
sys.path.insert(0, str(RUNTIME))
module_spec = importlib.util.spec_from_file_location("v149_training", RUNTIME/"run_general_ood_v147_randomized_recovery.py")
training = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = training
module_spec.loader.exec_module(training)

backend, np, torch = training.backend, training.np, training.torch
REVISION = "general_ood_v149_warm_recovery"
SOURCE_STEPS = 600000
ARMS = training.ARMS


def stage_spec(arm, smoke=False):
    extra = 512 if smoke else 100000
    return dict(training.stage_spec(arm, smoke), revision=REVISION, source_profile="no-pulse",
        source_steps=SOURCE_STEPS, new_steps=extra, steps=SOURCE_STEPS+extra,
        milestones=[SOURCE_STEPS+256, SOURCE_STEPS+512] if smoke else [650000, 700000],
        initialization="full-state-including-normal-replay-optimizers-and-rngs",
        schedule_assignment="subsequent-episode-resets-only")


def optimizer_steps(optimizer):
    return sorted({int(state["step"].item()) for state in optimizer.state.values()})


def warm_start(checkpoint, env, source_spec, seed, device):
    assert source_spec["profile"] == "no-pulse"
    model, old_progress = training.load_progress(checkpoint, env, source_spec, seed, device)
    assert env.onset is None and env.active_profile_name == "id" and env.stats["pulse_assigned_episodes"] == 0
    assert env._step == env.env._elapsed_steps
    private = np.random.RandomState(147000+seed)
    np.testing.assert_equal(env.pulse_rng.get_state(), private.get_state())
    receipt = dict(step=model.num_timesteps, optimizer_updates=model._n_updates,
        replay_size=model.replay_buffer.size(), replay_position=model.replay_buffer.pos,
        replay_capacity=model.replay_buffer.buffer_size, replay_full=model.replay_buffer.full,
        optimizer_steps=dict(actor=optimizer_steps(model.actor.optimizer), critic=optimizer_steps(model.critic.optimizer),
            entropy=optimizer_steps(model.ent_coef_optimizer)), log_entropy_coefficient=float(model.log_ent_coef.item()),
        reset_index=model.env.envs[0].env.resets, episode=env._episode, episode_step=env._step,
        onset=env.onset, active_profile=env.active_profile_name, source_coverage=dict(env.stats))
    assert receipt["step"] == old_progress["step"]
    assert all(steps == [model._n_updates] for steps in receipt["optimizer_steps"].values())
    return model, receipt


def validate_source(source, seed):
    assert (source["revision"], source["training_seed"], source["training_profile"], source["smoke"]) == (training.REVISION, seed, "no-pulse", False)
    assert source["config"] == backend.SETTINGS and source["spec"] == training.stage_spec("no-pulse")
    assert source["cumulative_training_interactions"] == SOURCE_STEPS and source["optimizer_updates"] == SOURCE_STEPS-255
    assert source["recovery_coverage"]["pulse_assigned_episodes"] == 0
    return dict(revision=source["revision"], seed=seed, profile="no-pulse", checkpoint=source["resumable_checkpoint"],
        training_interactions=SOURCE_STEPS, optimizer_updates=SOURCE_STEPS-255, config=source["config"])


def coverage_since_source(env, initial):
    return {key: count-initial.get(key, 0) for key, count in env.stats.items()}


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = json.loads(args.source_json.read_text())
    metadata = validate_source(source, args.seed)
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=backend.gym.__version__,
        mujoco=__import__("mujoco").__version__, stable_baselines3=backend.sb3.__version__)
    assert all(source["runtime"][k] == v for k, v in versions.items())
    env = training.RandomPulseGravity(args.profile, args.seed)
    spec, started = stage_spec(args.profile, args.smoke), time.monotonic()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = args.checkpoint_dir/"latest.pt"
    if args.resume_from:
        model, progress = training.load_progress(args.resume_from, env, spec, args.seed, args.device)
        assert progress["source"] == metadata
    else:
        checkpoint = source.get("initialization_checkpoint", source["resumable_checkpoint"])
        model, receipt = warm_start(checkpoint, env, training.stage_spec("no-pulse"), args.seed, args.device)
        assert receipt["step"] == receipt["replay_size"] == receipt["replay_position"] == SOURCE_STEPS
        assert receipt["optimizer_updates"] == SOURCE_STEPS-255 and receipt["replay_capacity"] == 1000000 and not receipt["replay_full"]
        assert receipt["source_coverage"] == source["recovery_coverage"]
        progress = dict(step=SOURCE_STEPS, source=metadata, initialization=receipt, blocks=[], periodic_evaluations=[], snapshots=[])
        progress["periodic_evaluations"].append(dict(step=SOURCE_STEPS, **backend.periodic_evaluation(model, "id", args.smoke)))
        training.save_progress(model, env, progress, spec, args.seed, latest)
    backend.bind_task_logger(model, args.checkpoint_dir)
    initial = model.num_timesteps
    print(f"[INIT] seed{args.seed} {args.profile}: source={SOURCE_STEPS} replay={progress['initialization']['replay_size']} current={initial} restored optimizers/RNGs", flush=True)
    while model.num_timesteps < spec["steps"]:
        before = model._n_updates
        model.learn(spec["block_steps"], reset_num_timesteps=False, log_interval=None)
        step = progress["step"] = model.num_timesteps
        assert step <= spec["steps"] and model._n_updates == step-255
        training.assert_physics(env)
        coverage = coverage_since_source(env, progress["initialization"]["source_coverage"])
        assert coverage.get("pulse_steps", 0)+coverage["baseline_steps"] == step-SOURCE_STEPS
        values = model.logger.name_to_value
        metrics = {k: float(values[k]) for k in ("train/actor_loss", "train/critic_loss", "train/ent_coef") if k in values}
        assert all(np.isfinite(v) for v in metrics.values())
        progress["blocks"].append(dict(step=step, interactions=spec["block_steps"], updates=model._n_updates-before,
            metrics=metrics, recovery_coverage=coverage))
        print(f"[TRAIN] seed{args.seed} {args.profile} {step-SOURCE_STEPS}/{spec['new_steps']} cumulative={step} updates={model._n_updates} pulses={coverage.get('pulse_exposed_episodes',0)}", flush=True)
        if (step-SOURCE_STEPS) % spec["evaluation_interval"] == 0 and step < spec["steps"]:
            progress["periodic_evaluations"].append(dict(step=step, **backend.periodic_evaluation(model, "id", args.smoke)))
        if step in spec["milestones"]:
            checkpoint = args.checkpoint_dir/f"step{step}.zip"
            model.save(checkpoint)
            progress["snapshots"].append(dict(step=step, optimizer_updates=model._n_updates, checkpoint=str(checkpoint),
                final_evaluation=training.final_evaluation(checkpoint, args.smoke, model.device)))
        if (step-SOURCE_STEPS) % spec["checkpoint_interval"] == 0 or step == spec["steps"]:
            training.save_progress(model, env, progress, spec, args.seed, latest)
    assert model.observation_space.shape == (11,) and model.replay_buffer.handle_timeout_termination
    raw = env.unwrapped
    result = dict(revision=REVISION, training_seed=args.seed, training_profile=args.profile, smoke=args.smoke,
        spec=spec, config=backend.SETTINGS, environment="Hopper-v4", training_device=args.device, source=metadata,
        initialization=progress["initialization"], actual_new_training_interactions=model.num_timesteps-initial,
        continuation_training_interactions=model.num_timesteps-SOURCE_STEPS, cumulative_training_interactions=model.num_timesteps,
        continuation_optimizer_updates=model._n_updates-(SOURCE_STEPS-255), optimizer_updates=model._n_updates,
        recovery_coverage=coverage_since_source(env, progress["initialization"]["source_coverage"]),
        reward_settings=dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight), forward_reward_weight=float(raw._forward_reward_weight)),
        reward_changes=[], healthy_termination_changes=[], motor_limit_changes=[], state_resets_at_switch=0,
        policy_receives_profile_labels=False, policy_receives_onset_clock=False, observation_dim=11,
        use_context=False, use_symbolic_constraints=False, reward_normalization=False,
        snapshots=progress["snapshots"], training=progress, checkpoint_selection="predeclared650k700k-not-best",
        resumable_checkpoint=str(latest), bootstrap_mask="terminated-only", replay_handle_timeout_termination=True,
        optimizer_updates_during_evaluation=0, runtime=dict(**versions,
            gpu=torch.cuda.get_device_name() if args.device == "cuda" else None),
        peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if args.device == "cuda" else 0.,
        seconds=time.monotonic()-started, scope="Two paired full-state warm continuations; standard-SAC environment calibration, not new independent training seeds or CS-BAPR contribution")
    model.env.close()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} {args.profile}: {spec['new_steps']} additional interactions; {model.num_timesteps} cumulative", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(5, 6), required=True)
    parser.add_argument("--profile", choices=ARMS, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
