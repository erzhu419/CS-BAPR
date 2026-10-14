"""Matched standard SAC with randomized short-pulse recovery-state coverage."""
import argparse
from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[0]/"experiments/general_ood_v147_randomized_recovery_20261014/runtime"
           if HERE.name == "scripts" else HERE)
sys.path.insert(0, str(RUNTIME))
import analyze_general_ood_v146_pulse_recovery as evaluation

backend, np, torch = evaluation.backend, evaluation.np, evaluation.torch
REVISION = "general_ood_v147_randomized_recovery"
ARMS = ("no-pulse", "randomized-recovery")
TRAINING_PULSE = dict(probability=.5, gravity=2., duration=10, onset_min=80, onset_max=320)


def stage_spec(arm, smoke=False):
    assert arm in ARMS
    return dict(backend.stage_spec("id", smoke), revision=REVISION, profile=arm,
        training_pulse=dict(TRAINING_PULSE, probability=0. if arm == "no-pulse" else .5),
        pulse_rng_seed="147000 + training seed", heldout_peak_gravity=evaluation.PEAK,
        heldout_durations=[25, 50], policy_receives_profile_labels=False,
        policy_receives_onset_clock=False, state_resets_at_switch=0)


class RandomPulseGravity(backend.v137.v135.ExtremeOODWrapper):
    def __init__(self, arm, seed):
        self.arm, self.pulse_rng = arm, np.random.RandomState(147000+seed)
        self.onset, self.stats, self.episode_peak_steps = None, Counter(), 0
        super().__init__(backend.gym.make("Hopper-v4"), profiles=[backend.v137.profile_settings("id"),
            dict(backend.v138.profile_settings("gravity-only"), gravity=2.)],
            config=backend.v137.v135.ExtremeOODConfig(schedule="pulse", profile_family="structured-component",
                include_sensor_noise=False, include_action_noise=False, include_reward_noise=False))

    def _initial_profile_for_episode(self):
        return self._coerce_profile("id")

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        selected = self.arm == "randomized-recovery" and self.pulse_rng.rand() < .5
        self.onset = int(self.pulse_rng.randint(80, 321)) if selected else None
        self.episode_peak_steps = 0
        self.stats["episode_resets"] += 1
        self.stats["pulse_assigned_episodes"] += int(selected)
        return result

    def _maybe_switch_before_step(self):
        if self.onset is not None and self._step == self.onset:
            self.set_profile("gravity-only")
        elif self.onset is not None and self._step == self.onset+10:
            self.set_profile("id")

    def step(self, action):
        result = super().step(action)
        _, _, term, trunc, info = result
        peak = info["ood_profile"] == "gravity-only"
        self.stats["pulse_steps" if peak else "baseline_steps"] += 1
        if peak:
            self.stats["pulse_exposed_episodes"] += int(self.episode_peak_steps == 0)
            self.episode_peak_steps += 1
        if self.onset is not None and self._step == self.onset+11:
            self.stats["pulse_restored_episodes"] += 1
        if self.onset is not None and self._step == self.onset+110 and not term:
            self.stats["healthy_recovered100"] += 1
        if term or trunc:
            self.stats["completed_episodes"] += 1
            if term:
                self.stats["physical_failures_after_pulse" if self.episode_peak_steps else "physical_failures_without_exposure"] += 1
            else:
                self.stats["healthy_time_limits_with_pulse" if self.episode_peak_steps else "healthy_time_limits_without_pulse"] += 1
        return result


def environment_state(env):
    return dict(native=backend.environment_state(env.env), step=env._step, episode=env._episode,
        onset=env.onset, active_profile=env.active_profile_name, pulse_rng=env.pulse_rng.get_state(),
        wrapper_rng=env._rng.get_state(), stats=deepcopy(env.stats), episode_peak_steps=env.episode_peak_steps)


def restore_environment(env, state):
    # Restore the active model parameters before MuJoCo reconstructs derived state.
    env.set_profile(state["active_profile"])
    backend.restore_environment(env.env, state["native"])
    env._step, env._episode, env.onset = state["step"], state["episode"], state["onset"]
    env.pulse_rng.set_state(state["pulse_rng"])
    env._rng.set_state(state["wrapper_rng"])
    env.stats, env.episode_peak_steps = deepcopy(state["stats"]), state["episode_peak_steps"]


def save_progress(model, env, progress, spec, seed, path):
    serialized = io.BytesIO()
    model.save(serialized)
    monitor = model.env.envs[0]
    seeded = monitor.env
    assert isinstance(seeded, backend.TrainingResets)
    payload = dict(revision=spec["revision"], spec=spec, seed=seed, model=serialized.getvalue(),
        replay=model.replay_buffer, environment=environment_state(env), resets=seeded.resets,
        monitor={k: deepcopy(getattr(monitor, k)) for k in ("rewards", "needs_reset", "total_steps", "episode_returns", "episode_lengths", "episode_times")},
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if model.device.type == "cuda" else None, progress=progress)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_progress(path, env, spec, seed, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert (payload["revision"], payload["spec"], payload["seed"]) == (spec["revision"], spec, seed)
    model = backend.SAC.load(io.BytesIO(payload["model"]), env=backend.TrainingResets(env, seed), device=device, force_reset=False)
    model.replay_buffer = payload["replay"]
    model.replay_buffer.device = torch.device(device)
    env.reset(seed=0)
    restore_environment(env, payload["environment"])
    monitor = model.env.envs[0]
    monitor.env.resets = payload["resets"]
    for k, value in payload["monitor"].items():
        setattr(monitor, k, value)
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(payload["cuda_rng"])
    assert model._last_obs is not None and model.num_timesteps == payload["progress"]["step"]
    return model, payload["progress"]


def final_evaluation(checkpoint, smoke, training_device):
    spec = dict(schedules=evaluation.schedules(smoke), max_steps=256 if smoke else 1000,
        reset_seeds=[154000] if smoke else list(range(154000, 154005)),
        action_noise_seeds=[137000] if smoke else [137000, 137001, 137002],
        peak_gravity=evaluation.PEAK, recovery_steps=100, device="cpu")
    with backend.frozen_rng(training_device):
        model = backend.SAC.load(checkpoint, device="cpu")
        assert model.replay_buffer.size() == 0
        model.replay_buffer = None
        model.policy.set_training_mode(False)
        before = model._n_updates
        episodes = [evaluation.evaluate_episode(model, "id", dict(condition=s["name"],
            mode="deterministic" if n is None else "sampled", reset_seed=r, noise_seed=n), s, spec["max_steps"])
            for s in spec["schedules"] for r in spec["reset_seeds"] for n in (None, *spec["action_noise_seeds"])]
        assert model._n_updates == before
    base = {(e["reset_seed"], e["noise_seed"]): e for e in episodes if e["condition"] == "no-pulse"}
    pairs = 0
    for schedule in spec["schedules"][1:]:
        onset = schedule["onset"]
        for e in (e for e in episodes if e["condition"] == schedule["name"]):
            b = base[e["reset_seed"], e["noise_seed"]]
            left = next((m for m in e["prefix_markers"] if m["completed_steps"] == onset), None)
            right = next((m for m in b["prefix_markers"] if m["completed_steps"] == onset), None)
            assert left == right
            if left is None:
                assert e["length"] == b["length"] and not e["recovery"]["exposed"]
            pairs += 1
    return dict(protocol=spec, episodes=episodes, prefix_pairs_verified=pairs, optimizer_updates=0,
        replay_loaded=False, device="cpu", evaluation_interactions=sum(e["length"] for e in episodes))


def assert_physics(env):
    raw = env.unwrapped
    factor = 2. if env.active_profile_name == "gravity-only" else 1.
    for name, values in (("body_mass", raw.model.body_mass), ("geom_friction", raw.model.geom_friction),
                         ("gravity", raw.model.opt.gravity)):
        np.testing.assert_array_equal(values, env._base[name]*(factor if name == "gravity" else 1.))
    assert raw._terminate_when_unhealthy and raw._healthy_reward == 1. and raw.dt == .008


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    env = RandomPulseGravity(args.profile, args.seed)
    spec, started = stage_spec(args.profile, args.smoke), time.monotonic()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = args.checkpoint_dir/"latest.pt"
    if args.resume_from:
        model, progress = load_progress(args.resume_from, env, spec, args.seed, args.device)
    else:
        model = backend.make_model(env, args.seed, args.device)
        progress = dict(step=0, blocks=[], periodic_evaluations=[], snapshots=[])
        progress["periodic_evaluations"].append(dict(step=0, **backend.periodic_evaluation(model, "id", args.smoke)))
    backend.bind_task_logger(model, args.checkpoint_dir)
    initial = model.num_timesteps
    while model.num_timesteps < spec["steps"]:
        before = model._n_updates
        model.learn(spec["block_steps"], reset_num_timesteps=False, log_interval=None)
        step = progress["step"] = model.num_timesteps
        assert step <= spec["steps"] and model._n_updates == max(0, step-255)
        assert_physics(env)
        values = model.logger.name_to_value
        metrics = {k: float(values[k]) for k in ("train/actor_loss", "train/critic_loss", "train/ent_coef") if k in values}
        assert all(np.isfinite(v) for v in metrics.values())
        assert env.stats["pulse_steps"]+env.stats["baseline_steps"] == step
        progress["blocks"].append(dict(step=step, interactions=spec["block_steps"], updates=model._n_updates-before,
            metrics=metrics, recovery_coverage=dict(env.stats)))
        print(f"[TRAIN] seed{args.seed} {args.profile} {step}/{spec['steps']} updates={model._n_updates} pulses={env.stats['pulse_exposed_episodes']}", flush=True)
        if step % spec["evaluation_interval"] == 0 and step < spec["steps"]:
            progress["periodic_evaluations"].append(dict(step=step, **backend.periodic_evaluation(model, "id", args.smoke)))
        if step in spec["milestones"]:
            checkpoint = args.checkpoint_dir/f"step{step}.zip"
            model.save(checkpoint)
            progress["snapshots"].append(dict(step=step, optimizer_updates=model._n_updates, checkpoint=str(checkpoint),
                final_evaluation=final_evaluation(checkpoint, args.smoke, model.device)))
        if step % spec["checkpoint_interval"] == 0 or step == spec["steps"]:
            save_progress(model, env, progress, spec, args.seed, latest)
    assert model.observation_space.shape == (11,) and model.replay_buffer.handle_timeout_termination
    raw = env.unwrapped
    result = dict(revision=REVISION, training_seed=args.seed, training_profile=args.profile, smoke=args.smoke,
        spec=spec, config=backend.SETTINGS, environment="Hopper-v4", training_device=args.device,
        actual_new_training_interactions=model.num_timesteps-initial, cumulative_training_interactions=model.num_timesteps,
        optimizer_updates=model._n_updates, recovery_coverage=dict(env.stats),
        physics=dict(body_mass=raw.model.body_mass.tolist(), geom_friction=raw.model.geom_friction.tolist(),
            baseline_gravity=env._base["gravity"].tolist(), active_gravity=raw.model.opt.gravity.tolist()),
        reward_settings=dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight), forward_reward_weight=float(raw._forward_reward_weight)),
        reward_changes=[], healthy_termination_changes=[], motor_limit_changes=[], state_resets_at_switch=0,
        policy_receives_profile_labels=False, policy_receives_onset_clock=False, observation_dim=11,
        use_context=False, use_symbolic_constraints=False, reward_normalization=False,
        snapshots=progress["snapshots"], training=progress, checkpoint_selection="predeclared300k600k-not-best",
        resumable_checkpoint=str(latest), bootstrap_mask="terminated-only", replay_handle_timeout_termination=True,
        optimizer_updates_during_evaluation=0, runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            gymnasium=backend.gym.__version__, mujoco=__import__("mujoco").__version__, stable_baselines3=backend.sb3.__version__,
            gpu=torch.cuda.get_device_name() if args.device == "cuda" else None),
        peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if args.device == "cuda" else 0.,
        seconds=time.monotonic()-started, scope="Recovery-state coverage control with unchanged standard SAC, not CS-BAPR mechanism attribution")
    model.env.close()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} {args.profile}: {model.num_timesteps} interactions / {model._n_updates} updates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(5, 6), required=True)
    parser.add_argument("--profile", choices=ARMS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
