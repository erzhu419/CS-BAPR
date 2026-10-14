"""Frozen native-gravity pulses with paired no-pulse recovery controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[0]/"experiments/general_ood_v146_pulse_recovery_20261014/runtime"
           if HERE.name == "scripts" else HERE)
sys.path.insert(0, str(RUNTIME))
import run_general_ood_v145_gravity_calibration as training

backend, np, torch = training.backend, training.backend.np, training.backend.torch
REVISION = "general_ood_v146_pulse_recovery"
CASES = tuple(f"seed{s}_{p}" for s in (3, 4) for p in ("id", "gravity-1p5"))
PEAK = backend.v138.profile_settings("gravity-only")["gravity"]
ONSETS, DURATIONS, RECOVERY_STEPS = (100, 300), (10, 25, 50), 100


def schedules(smoke=False):
    result = [dict(name="no-pulse", onset=None, duration=None)]
    for onset in ((100,) if smoke else ONSETS):
        for duration in ((10, None) if smoke else (*DURATIONS, None)):
            result.append(dict(name=f"after{onset}_"+("permanent" if duration is None else f"pulse{duration}"),
                               onset=onset, duration=duration))
    return result


class PulseGravity(backend.v137.v135.ExtremeOODWrapper):
    def __init__(self, profile, schedule):
        base = backend.v137.profile_settings("id") if profile == "id" else training.profile_settings(profile)
        self.pulse, self.base_name = schedule, profile
        super().__init__(backend.gym.make("Hopper-v4"), profiles=[base, backend.v138.profile_settings("gravity-only")],
            config=backend.v137.v135.ExtremeOODConfig(schedule="pulse", profile_family="structured-component", static_profile=profile,
                include_sensor_noise=False, include_action_noise=False, include_reward_noise=False))

    def _initial_profile_for_episode(self):
        return self._coerce_profile(self.base_name)

    def _maybe_switch_before_step(self):
        onset, duration = self.pulse["onset"], self.pulse["duration"]
        if onset is not None and self._step == onset:
            self.set_profile("gravity-only")
        elif onset is not None and duration is not None and self._step == onset+duration:
            self.set_profile(self.base_name)


def recovery(episode, schedule):
    onset, duration = schedule["onset"], schedule["duration"]
    exposed = episode["profile_counts"].get("gravity-only", 0) > 0
    horizon = onset+duration+RECOVERY_STEPS if onset is not None and duration is not None else None
    evaluable = exposed and horizon is not None and (episode["length"] >= horizon or episode["physical_termination"])
    survived = horizon is not None and (episode["length"] > horizon or (episode["length"] == horizon and not episode["physical_termination"]))
    return dict(exposed=exposed, peak_steps=episode["profile_counts"].get("gravity-only", 0),
        recovery_horizon=horizon, recovery_evaluable=bool(evaluable), recovered=bool(evaluable and survived),
        baseline_restored=any(e["step"] > 1 and e["profile"] != "gravity-only" for e in episode["profile_events"]))


@torch.no_grad()
def evaluate_episode(model, profile, cell, schedule, limit):
    env = PulseGravity(profile, schedule)
    noise = backend.evaluation.noise_schedule(cell["noise_seed"], limit, 3)
    scale = 0. if cell["mode"] == "deterministic" else 1.
    rewards, velocities, stds, differences, events, markers = [], [], [], [], [], []
    counts = Counter()
    try:
        state, _ = env.reset(seed=cell["reset_seed"])
        raw = env.unwrapped
        dt = float(raw.dt)
        assert raw._terminate_when_unhealthy and raw._healthy_reward == 1.
        for step in range(limit):
            if step in ONSETS:
                markers.append(dict(completed_steps=step, observation=state.tolist(),
                    cumulative_return=float(np.sum(rewards)), qpos=raw.data.qpos.tolist(), qvel=raw.data.qvel.tolist()))
            tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, log_std, _ = model.actor.get_action_dist_params(tensor)
            std = log_std.exp()
            deterministic = model.policy.unscale_action(mean.tanh().squeeze(0).numpy())
            scaled = (mean+scale*std*noise[step:step+1]).tanh().squeeze(0).numpy()
            action = model.policy.unscale_action(scaled)
            executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
            state, reward, term, trunc, info = env.step(executed)
            velocity = float(info["x_velocity"])
            np.testing.assert_allclose(reward, 1.+raw._forward_reward_weight*velocity-raw._ctrl_cost_weight*float(np.square(executed).sum()), rtol=1e-9, atol=1e-9)
            active = info["ood_profile"]
            counts[active] += 1
            if not events or events[-1]["profile"] != active:
                events.append(dict(step=step+1, profile=active, actual_multipliers=backend.v137.v135.actual_multipliers(env)))
            rewards.append(float(reward)); velocities.append(velocity); stds.append(float(std.mean()))
            differences.append(float(np.linalg.norm(action-deterministic)))
            if term or trunc:
                break
        row = dict(**cell, context_mode="not-applicable", context_updates=0, noise_scale=scale,
            length=len(rewards), episode_return=float(np.sum(rewards)), mean_velocity=float(np.mean(velocities)),
            physical_termination=bool(term), time_limit_truncation=bool(trunc), diagnostic_cutoff=not(term or trunc),
            mean_policy_std=float(np.mean(stds)), mean_action_noise_l2=float(np.mean(differences)),
            profile_counts=dict(counts), non_id_steps=sum(n for p, n in counts.items() if p != "id"),
            profile_events=events, switches=len(events)-1, final_physical_state=backend.evaluation.v124.physical_state(raw),
            native_reward_verified=True, dt=dt, prefix_markers=markers)
        return dict(**row, recovery=recovery(row, schedule))
    finally:
        env.close()


def load_source(source, case):
    seed, profile = int(case[4]), case[6:]
    revision = backend.REVISION if profile == "id" else training.REVISION
    assert (source["revision"], source["training_seed"], source["training_profile"], source["smoke"]) == (revision, seed, profile, False)
    assert source["config"] == backend.SETTINGS and source["cumulative_training_interactions"] == 600000 and source["optimizer_updates"] == 599745
    snapshot = next(s for s in source["snapshots"] if s["step"] == 600000)
    model = backend.SAC.load(source.get("evaluation_checkpoint", snapshot["checkpoint"]), device="cpu")
    assert model._n_updates == 599745 and model.observation_space.shape == (11,) and model.replay_buffer.size() == 0
    model.replay_buffer = None
    model.policy.set_training_mode(False)
    model.policy.requires_grad_(False)
    return model, dict(revision=revision, seed=seed, profile=profile, checkpoint=snapshot["checkpoint"],
        training_interactions=600000, optimizer_updates=599745, config=source["config"])


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = json.loads(args.source_json.read_text())
    model, metadata = load_source(source, args.case)
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        gymnasium=backend.gym.__version__, mujoco=__import__("mujoco").__version__, stable_baselines3=backend.sb3.__version__)
    assert all(source["runtime"][k] == v for k, v in versions.items())
    spec = dict(case=args.case, schedules=schedules(args.smoke), smoke=args.smoke, max_steps=256 if args.smoke else 1000,
        reset_seeds=[154000] if args.smoke else list(range(154000, 154005)),
        action_noise_seeds=[137000] if args.smoke else [137000, 137001, 137002],
        peak_gravity=PEAK, recovery_steps=RECOVERY_STEPS, device="cpu")
    started = time.monotonic()
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, runtime=versions, source=metadata, episodes=[])
    assert (progress["revision"], progress["spec"], progress["source"], progress["runtime"]) == (REVISION, spec, metadata, versions)
    cells = [dict(condition=s["name"], mode="deterministic" if n is None else "sampled", reset_seed=r, noise_seed=n)
        for s in spec["schedules"] for r in spec["reset_seeds"] for n in (None, *spec["action_noise_seeds"])]
    assert [{k: e[k] for k in cells[0]} for e in progress["episodes"]] == cells[:len(progress["episodes"])]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    by_name = {s["name"]: s for s in spec["schedules"]}
    for cell in cells[len(progress["episodes"]):]:
        progress["episodes"].append(evaluate_episode(model, metadata["profile"], cell, by_name[cell["condition"]], spec["max_steps"]))
        temporary = args.checkpoint_dir/"progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.checkpoint_dir/"progress.json")
        print(f"[EVAL] {args.case} {len(progress['episodes'])}/{len(cells)} {cell['condition']} length={progress['episodes'][-1]['length']}", flush=True)
    # Compare the pre-intervention state and return, retaining unexposed episodes.
    base = {(e["reset_seed"], e["noise_seed"]): e for e in progress["episodes"] if e["condition"] == "no-pulse"}
    matches = 0
    for e in progress["episodes"]:
        onset = by_name[e["condition"]]["onset"]
        if onset is None:
            continue
        b = base[e["reset_seed"], e["noise_seed"]]
        left = next((m for m in e["prefix_markers"] if m["completed_steps"] == onset), None)
        right = next((m for m in b["prefix_markers"] if m["completed_steps"] == onset), None)
        assert left == right
        if left is None:
            assert e["length"] == b["length"] and not e["recovery"]["exposed"]
        matches += 1
    assert model._n_updates == 599745 and model.replay_buffer is None and not torch.cuda.is_initialized()
    result = dict(**progress, case=args.case, smoke=args.smoke, evaluation_device="cpu", cuda_initialized=False,
        optimizer_updates=0, new_training_interactions=0, replay_loaded=False, prefix_pairs_verified=matches,
        reward_changes=[], healthy_termination_changes=[], motor_limit_changes=[], state_resets_at_switch=0,
        seconds=time.monotonic()-started, evaluation_interactions=sum(e["length"] for e in progress["episodes"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"pulse_recovery.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {args.case}: {len(cells)} episodes; zero updates; paired prefixes={matches}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
