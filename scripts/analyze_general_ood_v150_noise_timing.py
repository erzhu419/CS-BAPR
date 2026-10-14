"""Oracle-clock action-noise attribution for frozen final700k SAC policies."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[0]/"experiments/general_ood_v150_noise_timing_20261014/runtime"
           if HERE.name == "scripts" else HERE)
sys.path.insert(0, str(RUNTIME))
module_spec = importlib.util.spec_from_file_location("v150_pulse_evaluation", RUNTIME/"analyze_general_ood_v146_pulse_recovery.py")
evaluation = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = evaluation
module_spec.loader.exec_module(evaluation)
backend, np, torch = evaluation.backend, evaluation.np, evaluation.torch
REVISION = "general_ood_v150_noise_timing"
SOURCE_REVISION = "general_ood_v149_warm_recovery"
CASES = tuple(f"seed{s}_{a}" for s in (5, 6) for a in ("no-pulse", "randomized-recovery"))
PROGRAMS = ("native", "global-quiet", "prefix-quiet", "pulse-quiet", "recovery-quiet", "pulse-recovery-quiet")
MARKERS = (100, 110, 210, 300, 310, 410)


def noise_scale(program, step, onset):
    if program == "native":
        return 1.
    if program == "global-quiet":
        return 0.
    end = onset+10
    quiet = {"prefix-quiet": step < onset,
             "pulse-quiet": onset <= step < end,
             "recovery-quiet": end <= step < end+100,
             "pulse-recovery-quiet": onset <= step < end+100}[program]
    return 0. if quiet else 1.


def stage_spec(case, smoke=False):
    schedules = [dict(name=f"{p}_no-pulse", program=p, clock_onset=None, onset=None, duration=None) for p in PROGRAMS[:2]]
    for onset in (100, 300):
        for pulse in (False, True):
            for program in (PROGRAMS[2:] if not pulse else PROGRAMS):
                schedules.append(dict(name=f"{program}_after{onset}_"+("pulse10" if pulse else "no-pulse"),
                    program=program, clock_onset=onset, onset=onset if pulse else None, duration=10 if pulse else None))
    return dict(case=case, smoke=smoke, schedules=schedules, max_steps=1000,
        reset_seeds=[155000] if smoke else list(range(155000, 155005)),
        action_noise_seeds=[139000] if smoke else [139000, 139001, 139002],
        peak_gravity=evaluation.PEAK, recovery_steps=100, device="cpu", source_step=700000,
        oracle_clock=True, policy_network_receives_clock=False, noise_draws="absolute-step-index-even-when-quiet")


def cells(spec):
    return [dict(condition=s["name"], mode="deterministic" if s["program"] == "global-quiet" else "sampled",
                 reset_seed=r, noise_seed=n)
            for s in spec["schedules"] for r in spec["reset_seeds"]
            for n in ([None] if s["program"] == "global-quiet" else spec["action_noise_seeds"])]


def source_metadata(source, case):
    seed, arm = int(case[4]), case[6:]
    assert (source["revision"], source["training_seed"], source["training_profile"], source["smoke"]) == (SOURCE_REVISION, seed, arm, False)
    assert source["config"] == backend.SETTINGS
    assert source["cumulative_training_interactions"] == 700000 and source["optimizer_updates"] == 699745
    assert source["continuation_training_interactions"] == source["continuation_optimizer_updates"] == 100000
    snapshot = next(s for s in source["snapshots"] if s["step"] == 700000)
    assert snapshot["optimizer_updates"] == 699745 and Path(snapshot["checkpoint"]).name == "step700000.zip"
    return dict(revision=SOURCE_REVISION, seed=seed, arm=arm, checkpoint=snapshot["checkpoint"],
        training_interactions=700000, optimizer_updates=699745, config=source["config"])


def load_source(source, case):
    metadata = source_metadata(source, case)
    model = backend.SAC.load(source.get("evaluation_checkpoint", metadata["checkpoint"]), device="cpu")
    assert model._n_updates == 699745 and model.observation_space.shape == (11,) and model.replay_buffer.size() == 0
    model.replay_buffer = None
    model.policy.set_training_mode(False)
    model.policy.requires_grad_(False)
    return model, metadata


@torch.no_grad()
def evaluate_episode(model, cell, schedule, limit):
    env = evaluation.PulseGravity("id", schedule)
    noise = backend.evaluation.noise_schedule(cell["noise_seed"], limit, 3)
    rewards, velocities, stds, effective_stds, differences, events, markers = [], [], [], [], [], [], []
    counts, scales = Counter(), Counter()
    try:
        state, _ = env.reset(seed=cell["reset_seed"])
        raw = env.unwrapped
        assert raw._terminate_when_unhealthy and raw._healthy_reward == 1.
        def mark(step):
            if step in MARKERS:
                markers.append(dict(completed_steps=step, observation=state.tolist(), cumulative_return=float(np.sum(rewards)),
                    qpos=raw.data.qpos.tolist(), qvel=raw.data.qvel.tolist()))
        for step in range(limit):
            mark(step)
            scale = noise_scale(schedule["program"], step, schedule["clock_onset"])
            tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, log_std, _ = model.actor.get_action_dist_params(tensor)
            std = log_std.exp()
            deterministic = model.policy.unscale_action(mean.tanh().squeeze(0).numpy())
            action = model.policy.unscale_action((mean+scale*std*noise[step:step+1]).tanh().squeeze(0).numpy())
            executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
            state, reward, term, trunc, info = env.step(executed)
            velocity = float(info["x_velocity"])
            np.testing.assert_allclose(reward, 1.+raw._forward_reward_weight*velocity-raw._ctrl_cost_weight*float(np.square(executed).sum()), rtol=1e-9, atol=1e-9)
            active = info["ood_profile"]
            counts[active] += 1
            scales[str(scale)] += 1
            if not events or events[-1]["profile"] != active:
                events.append(dict(step=step+1, profile=active, actual_multipliers=backend.v137.v135.actual_multipliers(env)))
            rewards.append(float(reward)); velocities.append(velocity); stds.append(float(std.mean()))
            effective_stds.append(scale*float(std.mean()))
            differences.append(float(np.linalg.norm(action-deterministic)))
            if term or trunc:
                break
        mark(len(rewards))
        row = dict(**cell, context_mode="not-applicable", context_updates=0, noise_program=schedule["program"],
            noise_scale=0. if schedule["program"] == "global-quiet" else (1. if schedule["program"] == "native" else None),
            noise_scale_counts=dict(scales), mean_effective_policy_std=float(np.mean(effective_stds)),
            length=len(rewards), episode_return=float(np.sum(rewards)), mean_velocity=float(np.mean(velocities)),
            physical_termination=bool(term), time_limit_truncation=bool(trunc), diagnostic_cutoff=not(term or trunc),
            mean_policy_std=float(np.mean(stds)), mean_action_noise_l2=float(np.mean(differences)),
            profile_counts=dict(counts), non_id_steps=sum(n for p, n in counts.items() if p != "id"),
            profile_events=events, switches=len(events)-1, final_physical_state=backend.evaluation.v124.physical_state(raw),
            native_reward_verified=True, dt=float(raw.dt), prefix_markers=markers)
        return dict(**row, recovery=evaluation.recovery(row, schedule))
    finally:
        env.close()


def prefix_pair(episode, baseline, boundary):
    left = next((m for m in episode["prefix_markers"] if m["completed_steps"] == boundary), None)
    right = next((m for m in baseline["prefix_markers"] if m["completed_steps"] == boundary), None)
    assert left == right
    if left is None:
        assert all(episode[k] == baseline[k] for k in ("length", "episode_return", "physical_termination", "final_physical_state"))


def validate_pairs(episodes, spec):
    rows = {(e["condition"], e["reset_seed"], e["noise_seed"]): e for e in episodes}
    physical_pairs = boundary_pairs = 0
    for e in episodes:
        schedule = next(s for s in spec["schedules"] if s["name"] == e["condition"])
        onset, program = schedule["clock_onset"], schedule["program"]
        if onset is None:
            continue
        if schedule["onset"] is not None:
            name = f"{program}_no-pulse" if program in PROGRAMS[:2] else e["condition"].replace("pulse10", "no-pulse")
            prefix_pair(e, rows[name, e["reset_seed"], e["noise_seed"]], onset)
            physical_pairs += 1
        pulse = schedule["onset"] is not None
        native_name = f"native_after{onset}_pulse10" if pulse else "native_no-pulse"
        if program == "prefix-quiet":
            prefix_pair(e, rows["global-quiet_no-pulse", e["reset_seed"], None], onset)
            boundary_pairs += 1
        elif program in PROGRAMS[3:]:
            prefix_pair(e, rows[native_name, e["reset_seed"], e["noise_seed"]], onset+10 if program == "recovery-quiet" else onset)
            boundary_pairs += 1
    return physical_pairs, boundary_pairs


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = json.loads(args.source_json.read_text())
    model, metadata = load_source(source, args.case)
    spec = stage_spec(args.case, args.smoke)
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=backend.gym.__version__,
        mujoco=__import__("mujoco").__version__, stable_baselines3=backend.sb3.__version__)
    assert all(source["runtime"][k] == v for k, v in versions.items())
    started = time.monotonic()
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(revision=REVISION, spec=spec, runtime=versions, source=metadata, episodes=[])
    assert (progress["revision"], progress["spec"], progress["source"], progress["runtime"]) == (REVISION, spec, metadata, versions)
    expected = cells(spec)
    assert [{k: e[k] for k in expected[0]} for e in progress["episodes"]] == expected[:len(progress["episodes"])]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    by_name = {s["name"]: s for s in spec["schedules"]}
    for cell in expected[len(progress["episodes"]):]:
        episode = evaluate_episode(model, cell, by_name[cell["condition"]], spec["max_steps"])
        progress["episodes"].append(episode)
        temporary = args.checkpoint_dir/"progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.checkpoint_dir/"progress.json")
        print(f"[EVAL] {args.case} {len(progress['episodes'])}/{len(expected)} {cell['condition']} length={episode['length']}", flush=True)
    pairs, boundaries = validate_pairs(progress["episodes"], spec)
    assert model._n_updates == 699745 and model.replay_buffer is None and not torch.cuda.is_initialized()
    result = dict(**progress, case=args.case, smoke=args.smoke, evaluation_device="cpu", cuda_initialized=False,
        optimizer_updates=0, new_training_interactions=0, replay_loaded=False, reused_reference_episodes=0,
        prefix_pairs_verified=pairs, noise_boundary_pairs_verified=boundaries, reward_changes=[], healthy_termination_changes=[],
        motor_limit_changes=[], state_resets_at_switch=0, seconds=time.monotonic()-started,
        evaluation_interactions=sum(e["length"] for e in progress["episodes"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"noise_timing.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {args.case}: {len(expected)} oracle diagnostic episodes; zero updates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
