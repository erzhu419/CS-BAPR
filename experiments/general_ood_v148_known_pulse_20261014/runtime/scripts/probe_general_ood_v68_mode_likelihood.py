"""Collect fresh labelled references, then replay the fixed v66 diagnostic cells."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import analyze_general_ood_v66_covariance_mix as v66
from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts.run_general_ood_v52 import SOURCE_NAME, restore_environment
from scripts.train_csbapr import MUJOCO_MODE_PROFILES, apply_mode_to_env
from scripts.v52_agent import ContinuationAgent
from scripts.v66_covariance_mix_readout import NAME as CONTROL
from scripts.v68_mode_likelihood import MODES, NAME, ModeLikelihoodReference, observe_series

REVISION = "general_ood_v68_mode_likelihood"
SPEC = dict(fit_seed=135000, density_calibration_seed=135010, normal_null_seed=135020, steps=6000)
BASE = v66.v65.v64.v63.v62
OBSERVERS = (CONTROL, NAME)
CHECKPOINT_STEPS = 1000


def reference_cells(spec):
    seeds = (spec["fit_seed"], spec["density_calibration_seed"], spec["normal_null_seed"])
    assert len(set(seeds)) == 3
    return [dict(role=role, seed=seed, mode=mode) for role, seed, modes in (
        ("fit", seeds[0], MODES), ("density_calibration", seeds[1], MODES),
        ("normal_null", seeds[2], ("normal",))) for mode in modes]


def trace_path(directory, cell):
    return directory/f"{cell['role']}_seed{cell['seed']}_{cell['mode']}.npz"


def collect_reference(agent, env, cell, steps, loop=None, checkpoint=None):
    profile = MUJOCO_MODE_PROFILES["Hopper-v4"][cell["mode"]]
    apply_mode_to_env(env, "Hopper-v4", profile)
    if loop is None:
        random.seed(cell["seed"])
        np.random.seed(cell["seed"])
        torch.manual_seed(cell["seed"])
        state, _ = env.reset(seed=cell["seed"])
        agent.reset_context()
        loop = dict(step=0, state=state, episode=0, initial_updates=agent.training_steps,
            rows={k: [] for k in v53.TRACE_KEYS})
    for step in range(loop["step"], steps):
        state = loop["state"]
        action = agent.select_action(state, deterministic=False)
        executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(executed)
        done = terminated or truncated
        agent.observe_context(state, action, reward, next_state, done)
        values = (state.copy(), action.copy(), executed.copy(), next_state.copy(), float(reward),
                  bool(done), bool(terminated), bool(truncated))
        for key, value in zip(v53.TRACE_KEYS, values):
            loop["rows"][key].append(value)
        loop["step"], loop["state"] = step+1, next_state
        if done and step+1 < steps:
            loop["episode"] += 1
            loop["state"], _ = env.reset(seed=cell["seed"]+10000*loop["episode"])
            agent.reset_context()
        if checkpoint is not None and (step+1) % CHECKPOINT_STEPS == 0:
            checkpoint(loop)
            print(f"[COLLECT] {cell['role']} {cell['mode']} {step+1}/{steps}", flush=True)
    raw = env.unwrapped.model
    np.testing.assert_array_equal(raw.body_mass, env._original_body_mass*profile["body_mass"])
    np.testing.assert_array_equal(raw.geom_friction, env._original_friction*profile["friction"])
    np.testing.assert_array_equal(raw.opt.gravity, env._original_gravity*profile["gravity"])
    assert agent.training_steps == loop["initial_updates"], "unexpected-policy-update"
    trace = {key: np.asarray(values) for key, values in loop["rows"].items()}
    record = dict(**cell, interactions=steps, optimizer_updates=0,
        physical_terminations=int(trace["terminated"].sum()), time_limit_truncations=int(trace["truncated"].sum()))
    return trace, record


def load_trace(path):
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


def window_statistics(trace, reference, mode, switch):
    series = observe_series(trace, reference)
    index = MODES.index(mode)
    truth = np.where(np.arange(len(trace["dones"])) < switch, 0, index)
    output = {}
    for window, (a, b) in v66.v65.v64.window_ranges(len(trace["dones"]), switch).items():
        likelihoods = series["mean_log_likelihoods"][a:b]
        target = truth[a:b]
        true_likelihoods = likelihoods[np.arange(b-a), target]
        others = likelihoods.copy()
        others[np.arange(b-a), target] = -np.inf
        competitor = np.max(others, axis=1)
        predicted = np.argmax(likelihoods, axis=1)
        output[window] = dict(mean_coefficient=float(series["risk"][a:b].mean()),
            mean_signal=float(series["signal"][a:b].mean()), mean_threshold=float(series["threshold"][a:b].mean()),
            mean_log_bayes_factor=float(series["log_bayes_factor"][a:b].mean()),
            mode_accuracy=float(np.mean(predicted == target)),
            mean_true_minus_best_other_log_likelihood=float(np.mean(true_likelihoods-competitor)),
            mean_true_minus_normal_log_likelihood=float(np.mean(true_likelihoods-likelihoods[:, 0])),
            mean_mode_weights={m: float(series["mode_weights"][a:b, i].mean()) for i, m in enumerate(MODES)},
            predicted_mode_fractions={m: float(np.mean(predicted == i)) for i, m in enumerate(MODES)})
    return output


def summarize(records):
    output = {}
    for group in ("v62_test", "v61_development_replay"):
        normal = [r for r in records if r["group"] == group and r["mode"] == "normal"]
        changed = [r for r in records if r["group"] == group and r["mode"] != "normal"]
        output[group] = {}
        for name in OBSERVERS:
            null = all(r["observers"][name][w]["mean_coefficient"] <= .1 for r in normal for w in BASE.NORMAL_WINDOWS)
            counts = {w: sum(r["change_minus_nochange"][name][w] > 0 for r in changed) for w in BASE.CHANGE_WINDOWS}
            output[group][name] = dict(usable_normal_null=null, positive_pairs=counts, changed_pairs=len(changed),
                largest_gated_normal_mean=max(r["observers"][name][w]["mean_coefficient"] for r in normal for w in BASE.NORMAL_WINDOWS),
                detection_gate_pass=null and all(v == len(changed) for v in counts.values()))
        output[group][NAME]["positive_true_mode_margin_pairs"] = {
            w: sum(r["observers"][NAME][w]["mean_true_minus_best_other_log_likelihood"] > 0 for r in changed)
            for w in BASE.CHANGE_WINDOWS}
    return output


def replay_cells(args, reference, replay, baseline):
    records, normal = [], None
    switch = baseline["stage_spec"]["switch_step"]
    for original in replay["records"]:
        cell = {k: original[k] for k in ("group", "seed", "updating", "mode")}
        if cell["group"] == "v62_test":
            kind = "updating" if cell["updating"] else "frozen"
            path = args.v62_dir/f"run_checkpoints/preflight/traces/test_seed{cell['seed']}_{kind}_{cell['mode']}.npz"
        else:
            assert cell["group"] == "v61_development_replay"
            path = args.v61_dir/f"run_checkpoints/preflight/traces/seed{cell['seed']}_{cell['mode']}.npz"
        trace = load_trace(path)
        statistics = window_statistics(trace, reference, cell["mode"], switch)
        observers = {CONTROL: deepcopy(original["observers"][CONTROL]), NAME: statistics}
        record = dict(**cell, saved_transitions=len(trace["dones"]), observers=observers)
        if cell["updating"]:
            if cell["mode"] == "normal":
                normal = record
            assert normal["group"] == cell["group"] and normal["seed"] == cell["seed"]
            record["change_minus_nochange"] = {name: {w: values["mean_coefficient"]-normal["observers"][name][w]["mean_coefficient"]
                for w, values in windows.items()} for name, windows in observers.items()}
            assert all(abs(record["change_minus_nochange"][NAME][w]) < 1e-10 for w in ("first800", "pre64"))
        records.append(record)
        print(f"[REPLAY] {cell['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def run(args):
    torch.set_num_threads(2)
    cells = reference_cells(SPEC)
    checkpoint_dir = args.checkpoint_dir
    stored = checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir/"preflight_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=SPEC, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == SPEC
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2", "cpu")
        agent.start_shared(args.source_dir/SOURCE_NAME, shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            apply_mode_to_env(env, "Hopper-v4", MUJOCO_MODE_PROFILES["Hopper-v4"][cell["mode"]])
            env.reset(seed=cell["seed"])
            restore_environment(env, extra["environment"])
        trace, record = collect_reference(agent, env, cell, SPEC["steps"], loop,
            lambda active: v53.save_progress(agent, env, progress, active, checkpoint))
        assert agent.replay_buffer.size == 6144, "reference-transitions-added-to-RL-replay"
        np.savez_compressed(trace_path(stored, cell), **trace)
        progress["records"].append(record)
        progress["next_cell"] = index+1
        v53.save_progress(agent, env, progress, None, checkpoint)
        env.close()
        del agent
        extra = None
    fitting = {c["mode"]: load_trace(trace_path(stored, c)) for c in cells if c["role"] == "fit"}
    density = {c["mode"]: load_trace(trace_path(stored, c)) for c in cells if c["role"] == "density_calibration"}
    nulls = [load_trace(trace_path(stored, c)) for c in cells if c["role"] == "normal_null"]
    reference = ModeLikelihoodReference.fit(fitting, density, nulls)
    metadata = deepcopy(reference.metadata())
    path = checkpoint_dir/"mode_likelihood_reference.pt"
    torch.save(dict(revision=REVISION, reference=reference, reference_metadata=metadata, spec=SPEC,
                    source_training_seed=1, source_checkpoint=SOURCE_NAME), path)
    print("[REFERENCE] Five known-mode models frozen before any evaluation replay", flush=True)
    baseline = json.loads((args.v62_dir/"run_results/preflight/causal_innovation.json").read_text())
    replay = json.loads((args.v66_dir/"run_results/screen/covariance_mix.json").read_text())
    assert baseline["revision"] == BASE.REVISION and baseline["stage_spec"] == BASE.SPEC
    assert replay["revision"] == v66.REVISION and len(replay["records"]) == 17
    assert replay["environment_interactions"] == replay["optimizer_updates"] == 0
    assert not {c["seed"] for c in cells} & {r["seed"] for r in replay["records"]}
    records = replay_cells(args, reference, replay, baseline)
    assert reference.metadata() == metadata
    summary = summarize(records)
    detection_pass = all(s[NAME]["detection_gate_pass"] for s in summary.values())
    identification_pass = all(all(n == s[NAME]["changed_pairs"] for n in s[NAME]["positive_true_mode_margin_pairs"].values())
                              for s in summary.values())
    output = dict(revision=REVISION, reference_spec=SPEC, source_training_seed=1, source_checkpoint=SOURCE_NAME,
        source_updates=99745, action_sampling="stochastic", episode_time_limit=1000,
        reference_records=progress["records"], environment_interactions=sum(r["interactions"] for r in progress["records"]),
        optimizer_updates=0, RL_replay_augmented=False, beta=-2., readout_controls_actor=False,
        known_mode_prior="five labelled simulator physics profiles; changed modes are not unseen to the observer",
        reference_metadata=metadata, server_reference_checkpoint=str(path), reference_frozen_during_replay=True,
        records=records, summary=summary, replayed_saved_transitions=sum(r["saved_transitions"] for r in records),
        detection_gate_pass=detection_pass, true_mode_margin_gate_pass=identification_pass,
        decision="retrospective-mode-likelihood-screen-pass-not-confirmation" if detection_pass and identification_pass else "hold-mode-likelihood-screen",
        limitations=["Extra labelled mode reference data and simulator profile access; not a normal-only detector or a matched-budget RL comparison.",
            "Frozen-policy reference collection; updating-policy drift is assessed only on reused development traces.",
            "One source training checkpoint and previously inspected evaluation traces; not independent confirmation.",
            "Tempered window likelihood weights are not sequential Bayesian belief or adaptive-policy return evidence."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir/"mode_likelihood.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "v61-dir", "v62-dir", "v66-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
