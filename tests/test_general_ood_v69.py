import argparse
from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v69_counterfactual as worker
from scripts.v68_mode_likelihood import MODES, NAME
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace


def close(env, shadows):
    for physical in (env, *shadows.values()):
        physical.close()


def test_contract_and_count_duplicate_normal_validation_steps():
    path = Path(__file__).resolve().parents[1]/"experiments/general_ood_v69_counterfactual_20261008/dispatch_manifest.json"
    m = json.loads(path.read_text())
    assert m["reference_spec"] == worker.SPEC
    cells = worker.reference_cells(worker.SPEC)
    assert len(cells) == m["reference_cells"] == 6
    assert 6*worker.SPEC["steps"] == m["normal_interactions"] == 36000
    assert 20*worker.SPEC["steps"] == m["counterfactual_step_calls"] == 120000
    assert m["normal_interactions"]+m["counterfactual_step_calls"] == m["added_interactions"] == 156000
    assert 4*worker.SPEC["steps"] == m["exact_normal_replay_checks"] == 24000
    assert 3*worker.SPEC["steps"] == m["added_optimizer_updates"] == 18000
    assert m["normal_windows"] == list(worker.v68.BASE.NORMAL_WINDOWS)
    assert m["change_windows"] == list(worker.v68.BASE.CHANGE_WINDOWS)
    assert not m["observer_settings_changed"] and not m["counterfactuals_added_to_RL_replay"]
    assert m["normal_null_streams_separate"] and not m["readout_controls_actor"]
    assert not {c["seed"] for c in cells} & set(worker.v68.BASE.SPEC["test"])


def test_full_state_replay_preserves_hidden_clipped_velocities_and_age():
    env = gym.make("Hopper-v4", max_episode_steps=3)
    worker.apply_mode_to_env(env, "Hopper-v4", worker.MUJOCO_MODE_PROFILES["Hopper-v4"]["normal"])
    env.reset(seed=135040)
    shadows = worker.make_shadows(env)
    raw = env.unwrapped
    velocity = raw.data.qvel.copy()
    velocity[0], velocity[-1] = 13., -12.
    raw.set_state(raw.data.qpos.copy(), velocity)
    state = raw._get_obs()
    assert np.max(np.abs(state[-6:])) == 10. and np.max(np.abs(velocity)) == 13.
    env._elapsed_steps = 2
    snapshot = worker.environment_state(env)
    action = np.asarray([.1, -.2, .3])
    actual = env.step(action)
    values = worker.shadow_steps(shadows, snapshot, state, action, actual)
    np.testing.assert_array_equal(values["normal"][0], actual[0])
    assert values["normal"][1:4] == actual[1:4] and values["normal"][3]
    assert any(not np.array_equal(values[m][0], actual[0]) for m in MODES[1:])
    close(env, shadows)


def fresh(shared_start):
    checkpoint, shared, _, _, _ = shared_start
    env = gym.make("Hopper-v4", max_episode_steps=3)
    shadows = worker.make_shadows(env)
    agent = worker.ContinuationAgent(env, "fixed-minus2", "cpu")
    agent.start_shared(checkpoint, shared)
    return agent, env, shadows


@pytest.mark.parametrize("updating", [False, True])
def test_matched_inputs_only_normal_replay_and_exact_resume(shared_start, tmp_path, monkeypatch, updating):
    cell = dict(role="fit", seed=135040, updating=updating)
    direct, env, shadows = fresh(shared_start)
    before = deepcopy(direct.actor.state_dict())
    normal, modes, record = worker.collect_reference(direct, env, shadows, cell, 8)
    for m in MODES:
        for key in ("states", "actions", "executed_actions"):
            np.testing.assert_array_equal(modes[m][key], normal[key])
    for key in worker.v68.v53.TRACE_KEYS:
        np.testing.assert_array_equal(modes["normal"][key], normal[key])
    assert record["environment_interactions"] == 48 and record["counterfactual_step_calls"] == 40
    assert record["exact_normal_replay_checks"] == 8 and record["optimizer_updates"] == (8 if updating else 0)
    if not updating:
        for key, value in before.items():
            assert torch.equal(value, direct.actor.state_dict()[key])
    close(env, shadows)
    partial, env, shadows = fresh(shared_start)
    path = tmp_path/"resume.pt"
    monkeypatch.setattr(worker, "CHECKPOINT_STEPS", 4)
    def save(loop):
        worker.v68.v53.save_progress(partial, env, {"cell": cell}, loop, path)
        raise RuntimeError("test interruption")
    with pytest.raises(RuntimeError, match="test interruption"):
        worker.collect_reference(partial, env, shadows, cell, 8, checkpoint=save)
    close(env, shadows)
    restored, env, shadows = fresh(shared_start)
    extra = restored.load(str(path))
    worker.apply_mode_to_env(env, "Hopper-v4", worker.MUJOCO_MODE_PROFILES["Hopper-v4"]["normal"])
    env.reset(seed=cell["seed"])
    worker.restore_step_state(env, extra["environment"])
    actual, recovered, resumed_record = worker.collect_reference(restored, env, shadows, cell, 8, extra["loop"])
    close(env, shadows)
    assert record == resumed_record
    for key in worker.v68.v53.TRACE_KEYS:
        np.testing.assert_array_equal(actual[key], normal[key])
        for m in MODES:
            np.testing.assert_array_equal(recovered[m][key], modes[m][key])
    assert direct.replay_buffer.size == restored.replay_buffer.size == shared_start[1]["replay"].size+(8 if updating else 0)
    for name in ("actor", "critic", "target_critic"):
        for key, value in getattr(direct, name).state_dict().items():
            assert torch.equal(value, getattr(restored, name).state_dict()[key])


def write_references(directory, spec):
    stored = directory/"traces"
    stored.mkdir(parents=True)
    records = []
    for cell in worker.reference_cells(spec):
        seed = cell["seed"]+int(cell["updating"])
        np.savez_compressed(worker.trace_path(stored, cell), **trace(seed))
        counterfactual = cell["role"] != "normal_null"
        if counterfactual:
            for mode in MODES:
                np.savez_compressed(worker.trace_path(stored, cell, mode), **trace(seed, mode))
        steps = spec["steps"]
        records.append(dict(**cell, normal_interactions=steps, counterfactual_step_calls=steps*5 if counterfactual else 0,
            environment_interactions=steps*6 if counterfactual else steps,
            labelled_mode_transitions=steps*5 if counterfactual else 0,
            exact_normal_replay_checks=steps if counterfactual else 0, optimizer_updates=steps if cell["updating"] else 0))
    return stored, records


def test_fitting_rejects_nonmatched_mode_inputs(tmp_path):
    spec = dict(worker.SPEC, steps=2000)
    stored, _ = write_references(tmp_path, spec)
    cells = worker.reference_cells(spec)
    reference = worker.fit_reference(stored, cells)
    for stats in reference.metadata()["bank"]["statistics"].values():
        assert stats["fit_transitions"] == stats["density_calibration_transitions"] == 4000
    assert reference.normal_null_calibration_transitions == 4000
    path = worker.trace_path(stored, cells[0], "slippery")
    data = worker.v68.load_trace(path)
    data["states"][0, 0] += .1
    np.savez_compressed(path, **data)
    with pytest.raises(AssertionError):
        worker.fit_reference(stored, cells)


def test_full_worker_keeps_both_controls_and_reference_budget(tmp_path, monkeypatch):
    spec = dict(worker.SPEC, steps=2000)
    monkeypatch.setattr(worker, "SPEC", spec)
    monkeypatch.setattr(worker.v68.BASE, "SPEC", dict(worker.v68.BASE.SPEC, steps=2000, switch_step=800))
    checkpoint_dir = tmp_path/"reference"
    _, records = write_references(checkpoint_dir, spec)
    resume = checkpoint_dir/"preflight_latest.pt"
    progress = dict(revision=worker.REVISION, spec=spec, next_cell=6, records=records)
    torch.save(dict(extra_state=dict(progress=progress, loop=None)), resume)
    v61, v62, v68, shared = [tmp_path/p for p in ("v61", "v62", "v68", "shared")]
    shared.mkdir()
    torch.save({}, shared/"shared.pt")
    (v62/"run_results/preflight").mkdir(parents=True)
    (v62/"run_results/preflight/causal_innovation.json").write_text(json.dumps(dict(
        revision=worker.v68.BASE.REVISION, stage_spec=worker.v68.BASE.SPEC)))
    cases = [dict(group="v62_test", seed=seed, mode=mode, updating=updating)
        for seed in worker.v68.BASE.SPEC["test"] for mode, updating in [("normal", False)]+[(m, True) for m in MODES]]
    cases += [dict(group="v61_development_replay", seed=132020, mode=m, updating=True) for m in MODES]
    previous_records, normal = [], None
    for cell in cases:
        if cell["group"] == "v62_test":
            kind = "updating" if cell["updating"] else "frozen"
            path = v62/f"run_checkpoints/preflight/traces/test_seed{cell['seed']}_{kind}_{cell['mode']}.npz"
        else:
            path = v61/f"run_checkpoints/preflight/traces/seed{cell['seed']}_{cell['mode']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **trace(cell["seed"], cell["mode"], switch=800))
        old_windows = {w: dict(mean_coefficient=.02 if cell["mode"] == "normal" else .03,
                              mean_true_minus_best_other_log_likelihood=.1)
            for w in worker.v68.v66.v65.v64.window_ranges(2000, 800)}
        old = dict(**cell, observers={worker.v68.CONTROL: deepcopy(old_windows), NAME: deepcopy(old_windows)})
        if cell["updating"]:
            if cell["mode"] == "normal":
                normal = old
            old["change_minus_nochange"] = {name: {w: d["mean_coefficient"]-normal["observers"][name][w]["mean_coefficient"]
                for w, d in windows.items()} for name, windows in old["observers"].items()}
        previous_records.append(old)
    previous = dict(revision=worker.v68.REVISION, records=previous_records, summary=worker.v68.summarize(previous_records))
    (v68/"run_results/screen").mkdir(parents=True)
    (v68/"run_results/screen/mode_likelihood.json").write_text(json.dumps(previous))
    args = argparse.Namespace(source_dir=tmp_path/"unused_source", shared_dir=shared, v61_dir=v61, v62_dir=v62,
        v68_dir=v68, checkpoint_dir=checkpoint_dir, result_dir=tmp_path/"results", resume_from=resume, device="cpu")
    worker.run(args)
    output = json.loads((args.result_dir/"counterfactual.json").read_text())
    assert output["normal_interactions"] == 12000 and output["counterfactual_step_calls"] == 40000
    assert output["environment_interactions"] == 52000 and output["optimizer_updates"] == 6000
    assert output["exact_normal_replay_checks"] == 8000 and output["labelled_mode_transitions"] == 40000
    assert output["replayed_saved_transitions"] == 34000 and len(output["records"]) == 17
    assert not output["counterfactuals_added_to_RL_replay"] and not output["readout_controls_actor"]
    for old, new in zip(previous_records, output["records"]):
        assert new["observers"][worker.v68.CONTROL] == old["observers"][worker.v68.CONTROL]
        assert new["observers"][worker.PREVIOUS] == old["observers"][NAME]
    for group in previous["summary"]:
        assert output["summary"][group][worker.PREVIOUS] == previous["summary"][group][NAME]
    saved = torch.load(output["server_reference_checkpoint"], map_location="cpu", weights_only=False)
    assert saved["reference_metadata"] == output["reference_metadata"] and saved["spec"] == spec
    assert not any(k in json.dumps(output["reference_metadata"]) for k in ('"coefficients"', '"precisions"', '"covariance"'))
