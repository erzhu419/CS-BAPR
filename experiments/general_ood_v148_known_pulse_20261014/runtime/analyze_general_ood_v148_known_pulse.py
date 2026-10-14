"""Frozen pulse amplitudes with same-run paired native controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[0]/"experiments/general_ood_v148_known_pulse_20261014/runtime"
           if HERE.name == "scripts" else HERE)
sys.path.insert(0, str(RUNTIME))
# Other frozen campaigns share this basename; bind this campaign's evaluator.
module_spec = importlib.util.spec_from_file_location("v148_pulse_evaluation", RUNTIME/"analyze_general_ood_v146_pulse_recovery.py")
evaluation = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = evaluation
module_spec.loader.exec_module(evaluation)
import run_general_ood_v147_randomized_recovery as training

backend, torch = evaluation.backend, evaluation.torch
REVISION = "general_ood_v148_known_pulse"
CASES = tuple(f"seed{s}_{a}" for s in (5, 6) for a in training.ARMS)
GRAVITY = 2.


class AmplitudePulse(evaluation.PulseGravity):
    def __init__(self, profile, schedule):
        assert profile == "id"
        super().__init__(profile, schedule)
        self._profile_by_name["gravity-only"]["gravity"] = schedule["gravity"]


def stage_spec(case, smoke=False):
    schedules = [dict(name="no-pulse", onset=None, duration=None, gravity=1.)]
    schedules += [dict(name=f"{name}_after{onset}_pulse10", onset=onset, duration=10, gravity=g)
        for name, g in (("g2p0", GRAVITY), ("g2p659", evaluation.PEAK)) for onset in (100, 300)]
    return dict(case=case, smoke=smoke, schedules=schedules,
        max_steps=1000, reset_seeds=[154000] if smoke else list(range(154000, 154005)),
        action_noise_seeds=[137000] if smoke else [137000, 137001, 137002],
        peak_gravities=[GRAVITY, evaluation.PEAK], recovery_steps=100, device="cpu")


def load_source(source, case):
    seed, arm = int(case[4]), case[6:]
    assert (source["revision"], source["training_seed"], source["training_profile"], source["smoke"]) == (training.REVISION, seed, arm, False)
    assert source["spec"] == training.stage_spec(arm) and source["config"] == backend.SETTINGS
    assert source["cumulative_training_interactions"] == 600000 and source["optimizer_updates"] == 599745
    snapshot = next(s for s in source["snapshots"] if s["step"] == 600000)
    assert snapshot["optimizer_updates"] == 599745 and snapshot["final_evaluation"]["device"] == "cpu"
    assert snapshot["final_evaluation"]["protocol"]["peak_gravity"] == evaluation.PEAK
    model = backend.SAC.load(source.get("evaluation_checkpoint", snapshot["checkpoint"]), device="cpu")
    assert model._n_updates == 599745 and model.observation_space.shape == (11,) and model.replay_buffer.size() == 0
    model.replay_buffer = None
    model.policy.set_training_mode(False)
    model.policy.requires_grad_(False)
    return model, dict(revision=training.REVISION, seed=seed, arm=arm, checkpoint=snapshot["checkpoint"],
        training_interactions=600000, optimizer_updates=599745, config=source["config"])


def prefix_pair(episode, baseline, onset):
    left = next((m for m in episode["prefix_markers"] if m["completed_steps"] == onset), None)
    right = next((m for m in baseline["prefix_markers"] if m["completed_steps"] == onset), None)
    assert left == right
    if left is None:
        assert episode["length"] == baseline["length"] and not episode["recovery"]["exposed"]


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = json.loads(args.source_json.read_text())
    model, metadata = load_source(source, args.case)
    spec = stage_spec(args.case, args.smoke)
    versions = dict(python=sys.version, torch=torch.__version__, numpy=backend.np.__version__,
        gymnasium=backend.gym.__version__, mujoco=__import__("mujoco").__version__, stable_baselines3=backend.sb3.__version__)
    assert all(source["runtime"][k] == v for k, v in versions.items())
    started = time.monotonic()
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, runtime=versions, source=metadata, episodes=[])
    assert (progress["revision"], progress["spec"], progress["source"], progress["runtime"]) == (REVISION, spec, metadata, versions)
    cells = [dict(condition=s["name"], mode="deterministic" if n is None else "sampled", reset_seed=r, noise_seed=n)
        for s in spec["schedules"] for r in spec["reset_seeds"] for n in (None, *spec["action_noise_seeds"])]
    assert [{k: e[k] for k in cells[0]} for e in progress["episodes"]] == cells[:len(progress["episodes"])]
    by_name = {s["name"]: s for s in spec["schedules"]}
    baseline = {(e["reset_seed"], e["noise_seed"]): e for e in progress["episodes"] if e["condition"] == "no-pulse"}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells[len(progress["episodes"]):]:
        schedule = by_name[cell["condition"]]
        episode = evaluation.evaluate_episode(model, "id", cell, schedule, spec["max_steps"], env_factory=AmplitudePulse)
        if schedule["onset"] is None:
            baseline[cell["reset_seed"], cell["noise_seed"]] = episode
        else:
            prefix_pair(episode, baseline[cell["reset_seed"], cell["noise_seed"]], schedule["onset"])
        progress["episodes"].append(episode)
        temporary = args.checkpoint_dir/"progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.checkpoint_dir/"progress.json")
        print(f"[EVAL] {args.case} {len(progress['episodes'])}/{len(cells)} {cell['condition']} length={episode['length']}", flush=True)
    assert model._n_updates == 599745 and model.replay_buffer is None and not torch.cuda.is_initialized()
    result = dict(**progress, case=args.case, smoke=args.smoke, evaluation_device="cpu", cuda_initialized=False,
        optimizer_updates=0, new_training_interactions=0, replay_loaded=False, reused_reference_episodes=0,
        prefix_pairs_verified=len(cells)-len(baseline), reward_changes=[], healthy_termination_changes=[], motor_limit_changes=[],
        state_resets_at_switch=0, seconds=time.monotonic()-started,
        evaluation_interactions=sum(e["length"] for e in progress["episodes"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"known_pulse.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {args.case}: {len(cells)} episodes; same-run paired controls; zero updates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
