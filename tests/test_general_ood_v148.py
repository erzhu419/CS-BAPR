import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v148_known_pulse as worker
from scripts import run_general_ood_v148_multinode as transport


@pytest.mark.parametrize("gravity", (2., worker.evaluation.PEAK))
def test_known_pulse_changes_only_peak_and_preserves_canonical_prefix(gravity):
    torch.set_num_threads(1)
    env, _ = worker.backend.physics_worker.make_env("id")
    model = worker.backend.make_model(env, 5, "cpu")
    cell = dict(condition="test", mode="sampled", reset_seed=154000, noise_seed=137000)
    no_pulse = dict(name="test", onset=None, duration=None, gravity=1.)
    canonical = worker.evaluation.evaluate_episode(model, "id", cell, no_pulse, 256)
    known = worker.evaluation.evaluate_episode(model, "id", cell, no_pulse, 256, env_factory=worker.AmplitudePulse)
    assert known == canonical
    wrapped = worker.AmplitudePulse("id", dict(name="test", onset=2, duration=10, gravity=gravity))
    wrapped.reset(seed=154000)
    for step in range(14):
        _, _, term, trunc, info = wrapped.step(np.zeros(3))
        assert not term and not trunc
        factor = gravity if 2 <= step < 12 else 1.
        np.testing.assert_array_equal(wrapped.unwrapped.model.opt.gravity, wrapped._base["gravity"]*factor)
        np.testing.assert_array_equal(wrapped.unwrapped.model.body_mass, wrapped._base["body_mass"])
        np.testing.assert_array_equal(wrapped.unwrapped.model.geom_friction, wrapped._base["geom_friction"])
        assert info["ood_profile"] == ("gravity-only" if factor != 1. else "id")
    assert worker.evaluation.PEAK == 2.6591479484724942
    wrapped.close(); model.env.close()


def test_reused_prefix_detects_real_mismatch_and_keeps_unexposed_failures():
    marker = dict(completed_steps=100, observation=[1.], cumulative_return=2., qpos=[3.], qvel=[4.])
    e = dict(prefix_markers=[marker], length=300, recovery=dict(exposed=True))
    worker.prefix_pair(e, dict(prefix_markers=[marker]), 100)
    with pytest.raises(AssertionError):
        worker.prefix_pair(e, dict(prefix_markers=[]), 100)
    e = dict(prefix_markers=[], length=80, recovery=dict(exposed=False))
    worker.prefix_pair(e, dict(prefix_markers=[], length=80), 100)


def test_transport_only_copies_final_zip_and_protocol_adds_two_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    row = dict(snapshots=[dict(step=300000, checkpoint="server/step300000.zip"), dict(step=600000, checkpoint="server/step600000.zip")], resumable_checkpoint="server/latest.pt")
    copies = []
    monkeypatch.setattr(transport.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(row)))
    monkeypatch.setattr(transport.transport, "copy_artifact", lambda p: copies.append(p) or tmp_path/"node/step600000.zip")
    path = transport.stage_source("seed5_no-pulse")
    assert copies == ["server/step600000.zip"]
    assert json.loads(path.read_text())["snapshots"] == row["snapshots"]
    assert worker.CASES == transport.CASES and len(worker.stage_spec(worker.CASES[0])["schedules"]) == 5
