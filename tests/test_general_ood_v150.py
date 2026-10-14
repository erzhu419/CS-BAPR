import json
from types import SimpleNamespace

import pytest
import torch

from scripts import analyze_general_ood_v150_noise_timing as worker
from scripts import run_general_ood_v150_multinode as transport


@pytest.mark.parametrize("onset", (100, 300))
def test_noise_windows_have_exact_boundaries(onset):
    points = (0, onset-1, onset, onset+9, onset+10, onset+109, onset+110, 999)
    expected = {
        "native": (1, 1, 1, 1, 1, 1, 1, 1),
        "global-quiet": (0, 0, 0, 0, 0, 0, 0, 0),
        "prefix-quiet": (0, 0, 1, 1, 1, 1, 1, 1),
        "pulse-quiet": (1, 1, 0, 0, 1, 1, 1, 1),
        "recovery-quiet": (1, 1, 1, 1, 0, 0, 1, 1),
        "pulse-recovery-quiet": (1, 1, 0, 0, 0, 0, 1, 1)}
    for program, values in expected.items():
        assert tuple(worker.noise_scale(program, t, onset) for t in points) == values
    assert worker.noise_scale("native", 0, None) == 1.
    assert worker.noise_scale("global-quiet", 0, None) == 0.


def test_protocol_has_fresh_repeats_and_no_redundant_deterministic_noise():
    spec = worker.stage_spec(worker.CASES[0])
    cells = worker.cells(spec)
    assert len(cells) == 300 and len(spec["schedules"]) == 22
    assert spec["reset_seeds"] == list(range(155000, 155005))
    assert spec["action_noise_seeds"] == [139000, 139001, 139002]
    assert spec["peak_gravity"] == 2.6591479484724942 and spec["source_step"] == 700000
    assert spec["oracle_clock"] and not spec["policy_network_receives_clock"]
    assert sum(c["noise_seed"] is None for c in cells) == 15
    assert len(worker.cells(worker.stage_spec(worker.CASES[0], True))) == 22


@pytest.mark.parametrize("program", ("native", "global-quiet"))
def test_unchanged_action_modes_match_original_evaluator(program):
    torch.set_num_threads(1)
    env, _ = worker.backend.physics_worker.make_env("id")
    model = worker.backend.make_model(env, 5, "cpu")
    spec = worker.stage_spec(worker.CASES[0], True)
    cell = next(c for c in worker.cells(spec) if c["condition"] == f"{program}_no-pulse")
    schedule = next(s for s in spec["schedules"] if s["name"] == cell["condition"])
    original = worker.evaluation.evaluate_episode(model, "id", cell, schedule, 128)
    current = worker.evaluate_episode(model, cell, schedule, 128)
    for key, value in original.items():
        if key != "prefix_markers":
            assert current[key] == value
    assert all(m in current["prefix_markers"] for m in original["prefix_markers"])
    assert current["noise_scale_counts"] == {str(0. if program == "global-quiet" else 1.): current["length"]}
    if program == "global-quiet":
        assert current["mean_effective_policy_std"] == current["mean_action_noise_l2"] == 0.
    model.env.close()


def test_prefix_checks_detect_changed_states_and_retain_pre_boundary_falls():
    marker = dict(completed_steps=100, observation=[1.], cumulative_return=2., qpos=[3.], qvel=[4.])
    worker.prefix_pair(dict(prefix_markers=[marker]), dict(prefix_markers=[marker]), 100)
    with pytest.raises(AssertionError):
        worker.prefix_pair(dict(prefix_markers=[marker]), dict(prefix_markers=[]), 100)
    early = dict(prefix_markers=[], length=80, episode_return=2., physical_termination=True, final_physical_state={"x": 1.})
    worker.prefix_pair(early, dict(early), 100)
    with pytest.raises(AssertionError):
        worker.prefix_pair(early, dict(early, episode_return=3.), 100)


def test_transport_selects_only_final700k_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    row = dict(snapshots=[dict(step=650000, checkpoint="server/step650000.zip"), dict(step=700000, checkpoint="server/step700000.zip")], resumable_checkpoint="server/latest.pt")
    copies = []
    monkeypatch.setattr(transport.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(row)))
    monkeypatch.setattr(transport.transport, "copy_artifact", lambda p: copies.append(p) or tmp_path/"node/step700000.zip")
    path = transport.stage_source(worker.CASES[0])
    assert copies == ["server/step700000.zip"]
    assert json.loads(path.read_text())["snapshots"] == row["snapshots"]
    assert worker.CASES == transport.CASES
