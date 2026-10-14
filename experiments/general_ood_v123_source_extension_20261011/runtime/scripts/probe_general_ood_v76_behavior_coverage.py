"""Matched frozen/updating reference coverage with a fixed mode observer."""
import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v74_density_coverage as v74
from scripts import analyze_general_ood_v75_support as v75
from scripts.v73_recursive_mode_readout import MODES, RecursiveModeReference, observe_likelihoods
from scripts.v76_behavior_coverage import BehaviorCoverageBank

v73, v70, v69, v68 = v74.v73, v74.v73.v70, v74.v69, v74.v68
REVISION = "general_ood_v76_behavior_coverage"
SPEC = dict(density_calibration=[135170, 135171], test=[135180, 135181], steps=6000, switch_step=3800)
CONDITIONS = ("v72", "v74_slippery", "frozen_mixed", "updating_mixed")
CHECKPOINT_STEPS = 1000


def reference_cells(spec):
    return [dict(role="density_extension", seed=s, mode=m, updating=u)
            for s in spec["density_calibration"] for u in (False, True) for m in ("normal", "slippery")]


def test_cells(spec):
    return [dict(role="test", seed=s, mode=m, updating=True) for s in spec["test"] for m in MODES]


def trace_path(directory, cell):
    return v69.trace_path(directory, cell, cell["mode"])


def check_seeds():
    calibration, tests = (set(SPEC[k]) for k in ("density_calibration", "test"))
    assert len(calibration) == len(tests) == 2 and not calibration & tests
    previous = set(v74.SPEC["density_calibration"]+v74.SPEC["test"]+v73.SPEC["test"]+v74.v72.SPEC["test"])
    previous |= set(v74.v71.SPEC["fit"]+v74.v71.SPEC["density_calibration"]+v74.v71.SPEC["test"])
    previous |= set(v70.SPEC["seeds"]) | set(v68.BASE.SPEC["test"]) | {132020, 135179}
    previous |= {spec[k] for spec in (v69.SPEC, v68.SPEC) for k in spec if k.endswith("seed")}
    assert not (calibration | tests) & previous


def collect_updating(agent, env, shadows, cell, spec, loop=None, checkpoint=None):
    assert cell["updating"] and tuple(shadows) == MODES
    if loop is None:
        random.seed(cell["seed"])
        np.random.seed(cell["seed"])
        torch.manual_seed(cell["seed"])
        v70.apply_mode(env, "normal")
        state, _ = env.reset(seed=cell["seed"])
        agent.reset_context()
        loop = dict(step=0, state=state, episode=0, initial_updates=agent.training_steps,
            initial_replay_size=agent.replay_buffer.size, q_loss_sum=0.,
            rows={k: [] for k in (*v68.v53.TRACE_KEYS, *v70.ORACLE_KEYS)},
            recent={k: deque(maxlen=agent.config.surprise_window) for k in ("state", "action", "reward", "context")})
    for step in range(loop["step"], spec["steps"]):
        mode = v70.active_mode(cell, step, spec)
        if step == spec["switch_step"]:
            v70.apply_mode(env, mode)
        state = loop["state"]
        action = agent.select_action(state, deterministic=False)
        context = agent._current_context_np().copy()
        executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
        snapshot = v69.environment_state(env)
        clipped = bool(np.any(np.abs(env.unwrapped.data.qvel) > 10.))
        actual = env.step(executed)
        actual_physical = v70.physical_state(env)
        observations, physical = [], []
        for candidate, shadow in shadows.items():
            v69.restore_step_state(shadow, snapshot)
            np.testing.assert_array_equal(shadow.unwrapped._get_obs(), state)
            values = shadow.step(executed)
            next_physical = v70.physical_state(shadow)
            if candidate == mode:
                np.testing.assert_array_equal(values[0], actual[0])
                np.testing.assert_array_equal(next_physical, actual_physical)
                assert values[1:4] == actual[1:4], "true-mode-shadow-not-exact"
            observations.append(values[0].copy())
            physical.append(next_physical)
        v69.append_transition(loop["rows"], state, action, executed, actual)
        for key, value in zip(v70.ORACLE_KEYS, (observations, physical, actual_physical,
                snapshot["values"], snapshot["elapsed_steps"], clipped)):
            loop["rows"][key].append(value)
        next_state, reward, terminated, truncated, _ = actual
        done = terminated or truncated
        for key, value in (("state", state.copy()), ("action", action.copy()), ("reward", float(reward)), ("context", context)):
            loop["recent"][key].append(value)
        # Only the real transition enters replay; the observer is evaluated offline.
        agent.store_transition(state, action, reward, next_state, float(done))
        metrics = agent.update(recent_rollout=v69._recent_rollout_payload(loop["recent"]))
        assert metrics is not None and metrics["effective_beta"] == -2.
        loop["q_loss_sum"] += float(metrics["q_loss"])
        loop["step"], loop["state"] = step+1, next_state
        if done and step+1 < spec["steps"]:
            loop["episode"] += 1
            loop["state"], _ = env.reset(seed=cell["seed"]+10000*loop["episode"])
            agent.reset_context()
        if checkpoint is not None and (step+1) % CHECKPOINT_STEPS == 0:
            checkpoint(loop)
            print(f"[COLLECT] {cell['role']} seed{cell['seed']} {cell['mode']} updating=True {step+1}/{spec['steps']}", flush=True)
    updates = agent.training_steps-loop["initial_updates"]
    assert updates == spec["steps"]
    assert agent.replay_buffer.size == loop["initial_replay_size"]+updates, "oracle-entered-RL-replay"
    profile = v69.MUJOCO_MODE_PROFILES["Hopper-v4"][cell["mode"]]
    raw = env.unwrapped.model
    np.testing.assert_array_equal(raw.body_mass, env._original_body_mass*profile["body_mass"])
    np.testing.assert_array_equal(raw.geom_friction, env._original_friction*profile["friction"])
    np.testing.assert_array_equal(raw.opt.gravity, env._original_gravity*profile["gravity"])
    trace = {k: np.asarray(v) for k, v in loop["rows"].items()}
    record = dict(**cell, actual_transitions=spec["steps"], counterfactual_step_calls=5*spec["steps"],
        environment_step_calls=6*spec["steps"], exact_true_mode_replays=spec["steps"], optimizer_updates=updates,
        physical_terminations=int(trace["terminated"].sum()), time_limit_truncations=int(trace["truncated"].sum()),
        clipped_input_velocities=int(trace["clipped_input_velocities"].sum()), mean_q_loss=loop["q_loss_sum"]/updates)
    return trace, record


def collect_cells(args, shared, progress, extra, cells, stop):
    stored = args.checkpoint_dir/"traces"
    checkpoint = args.checkpoint_dir/"latest.pt"
    for index in range(progress["next_cell"], stop):
        cell = cells[index]
        env = gym.make("Hopper-v4")
        shadows = v69.make_shadows(env)
        agent = v69.ContinuationAgent(env, "fixed-minus2", args.device)
        agent.start_shared(args.source_dir/v69.SOURCE_NAME, shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            v70.apply_mode(env, v70.active_mode(cell, max(0, loop["step"]-1), SPEC))
            env.reset(seed=cell["seed"])
            v69.restore_step_state(env, extra["environment"])
        collector = collect_updating if cell["updating"] else v70.collect
        trace, record = collector(agent, env, shadows, cell, SPEC, loop,
            lambda active: v68.v53.save_progress(agent, env, progress, active, checkpoint))
        np.savez_compressed(trace_path(stored, cell), **trace)
        if cell["mode"] != "normal":
            normal = v68.load_trace(trace_path(stored, dict(cell, mode="normal")))
            assert v68.v53.paired_prefix_equal(trace, normal, SPEC["switch_step"]), "non-matched-prefix"
        progress["records"].append(record)
        progress["next_cell"] = index+1
        v68.v53.save_progress(agent, env, progress, None, checkpoint)
        for physical in (env, *shadows.values()):
            physical.close()
        del agent
        torch.cuda.empty_cache()
        extra = None
    return extra


def extensions(stored, spec, updating):
    cells = [c for c in reference_cells(spec) if c["updating"] == updating]
    traces = [v68.load_trace(trace_path(stored, c)) for c in cells]
    suffix = slice(spec["switch_step"], spec["steps"], 2)
    return {m: {key: np.concatenate([t["oracle_observations"][suffix, i] if key == "next_states"
                                    else t[key][suffix] for t in traces]) for key in v74.KEYS}
            for i, m in enumerate(MODES)}


def freeze_references(args, stored):
    saved = torch.load(args.v74_dir/"run_checkpoints/preflight/density_coverage_references.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v74.REVISION and saved["spec"] == v74.SPEC
    previous = saved["references"]
    original = previous["v72"]["recursive"].bank
    assert original.density_transitions == v74.BASE_DENSITY_TRANSITIONS
    old = args.v69_dir/"run_checkpoints/reference/traces"
    nulls = [v68.load_trace(v69.trace_path(old, c)) for c in v69.reference_cells(v69.SPEC) if c["role"] == "normal_null"]
    references = dict(v72=previous["v72"]["recursive"], v74_slippery=previous["slippery_state"]["recursive"])
    for condition, updating in (("frozen_mixed", False), ("updating_mixed", True)):
        bank = BehaviorCoverageBank.extend(original, extensions(stored, SPEC, updating), condition)
        assert bank.added_density_transitions == len(SPEC["density_calibration"])*(SPEC["steps"]-SPEC["switch_step"])
        reference = RecursiveModeReference.calibrate(bank, nulls)
        assert reference.normal_null_calibration_transitions == references["v72"].normal_null_calibration_transitions
        references[condition] = reference
        print(f"[DENSITY] {condition}: +{bank.added_density_transitions}/mode; original forward/scales and recursive rule unchanged", flush=True)
    return references


def evaluation_inputs(args, stored):
    output = [(dict(group="v76_fresh_updating", seed=c["seed"], mode=c["mode"], updating=True), trace_path(stored, c))
              for c in test_cells(SPEC)]
    for cell, path in v74.evaluation_inputs(args, args.v74_dir/"run_checkpoints/preflight/traces"):
        if cell["group"] == "v74_fresh_frozen":
            cell = dict(cell, group="v74_development_replay")
        output.append((cell, path))
    return output


def evaluate(args, stored, references):
    records = []
    for cell, path in evaluation_inputs(args, stored):
        trace = v68.load_trace(path)
        observers, support = {}, {}
        for condition, reference in references.items():
            likelihoods = reference.bank.log_likelihoods(trace)
            observers[condition] = v73.window_statistics(trace, likelihoods,
                observe_likelihoods(likelihoods, trace["dones"], reference, "recursive"), cell["mode"])
            support[condition] = {w: v75.support({k: trace[k][a:b] for k in v74.KEYS}, reference.bank,
                references["v72"].bank.density_transitions) for w, a, b in (
                ("first_post50", SPEC["switch_step"], SPEC["switch_step"]+50), ("last64", SPEC["steps"]-64, SPEC["steps"]))}
        record = dict(**cell, observers=observers, support=support)
        normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == cell["group"]
            and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = {c: {w: d["mean_coefficient"]-normal["observers"][c][w]["mean_coefficient"]
            for w, d in windows.items()} for c, windows in observers.items()}
        records.append(record)
        print(f"[EVAL] {cell['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def summarize(records):
    output = {}
    for group in dict.fromkeys(r["group"] for r in records):
        normal = [r for r in records if r["group"] == group and r["mode"] == "normal"]
        changed = [r for r in records if r["group"] == group and r["mode"] != "normal"]
        output[group] = {}
        for condition in CONDITIONS:
            maximum = max(r["observers"][condition][w]["mean_coefficient"] for r in normal for w in v68.BASE.NORMAL_WINDOWS)
            contrasts = {w: sum(r["change_minus_nochange"][condition][w] > 0 for r in changed) for w in v68.BASE.CHANGE_WINDOWS}
            margins = {w: sum(r["observers"][condition][w]["mean_true_minus_best_other_score"] > 0 for r in changed)
                       for w in v68.BASE.CHANGE_WINDOWS}
            output[group][condition] = dict(changed_pairs=len(changed), largest_normal_mean=maximum,
                positive_risk_contrast_pairs=contrasts, positive_true_mode_margin_pairs=margins,
                mean_changed_accuracy={w: float(np.mean([r["observers"][condition][w]["mode_accuracy"] for r in changed]))
                    for w in v68.BASE.CHANGE_WINDOWS},
                screen_pass=maximum <= .1 and all(n == len(changed) for n in (*contrasts.values(), *margins.values())))
    return output


def run(args):
    torch.set_num_threads(2)
    check_seeds()
    assert all(SPEC[k] == v73.SPEC[k] for k in ("steps", "switch_step"))
    stored = args.checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=SPEC, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == SPEC
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    refs = reference_cells(SPEC)
    cells = refs+test_cells(SPEC)
    extra = collect_cells(args, shared, progress, extra, cells, len(refs))
    destination = args.checkpoint_dir/"behavior_coverage_references.pt"
    if destination.is_file():
        saved = torch.load(destination, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == SPEC
        references = saved["references"]
    else:
        references = freeze_references(args, stored)
        torch.save(dict(revision=REVISION, spec=SPEC, references=references), destination)
    print("[REFERENCE] Four density banks with the unchanged recursive readout frozen BEFORE fresh updating tests", flush=True)
    metadata = {c: deepcopy(r.metadata()) for c, r in references.items()}
    collect_cells(args, shared, progress, extra, cells, len(cells))
    records = evaluate(args, stored, references)
    assert metadata == {c: r.metadata() for c, r in references.items()}
    summary = summarize(records)
    totals = {k: sum(r[k] for r in progress["records"]) for k in ("actual_transitions", "counterfactual_step_calls",
        "environment_step_calls", "exact_true_mode_replays", "optimizer_updates")}
    assert totals["actual_transitions"] == len(cells)*SPEC["steps"]
    assert totals["optimizer_updates"] == sum(c["updating"] for c in cells)*SPEC["steps"]
    passed = all(group["updating_mixed"]["screen_pass"] for group in summary.values())
    output = dict(revision=REVISION, spec=SPEC, source_training_seed=1, source_updates=99745, collection_device=args.device,
        reference_records=progress["records"][:len(refs)], test_records=progress["records"][len(refs):], **totals,
        reference_labels_per_mode_per_condition=len(SPEC["density_calibration"])*(SPEC["steps"]-SPEC["switch_step"]),
        profile_mixture="equal normal/slippery", suffix_sampling="every second post-switch transition",
        forward_models_refitted=False, scaling_refitted=False, conditional_density_refitted=True,
        temporal_rules_changed=False, readout_controls_actor=False, counterfactuals_added_to_RL_replay=False,
        reference_metadata={c: {k: v for k, v in m.items() if k != "phase"} for c, m in metadata.items()},
        server_reference_checkpoint=str(destination), records=records, summary=summary,
        decision="behavior-coverage-screen-pass-not-RL-confirmation" if passed else "hold-behavior-coverage",
        limitations=["One source training seed, mild known modes and privileged offline counterfactual labels.",
            "Frozen/updating controls match label budgets, profiles, times and device; updating reference collection additionally uses optimizer updates.",
            "Density mean and covariance support change together; this is not an actor-feedback or return comparison.",
            "Previously inspected paths remain development replays, including earlier CPU-collected frozen paths."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"behavior_coverage.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    run(parser.parse_args())
