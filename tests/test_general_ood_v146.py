from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v146_pulse_recovery as worker
from scripts import run_general_ood_v146_multinode as transport
from scripts import run_general_ood_v144_fixed_profile_sac as logging_backend


@pytest.mark.parametrize("profile", ("id", "gravity-1p5"))
@pytest.mark.parametrize("noise", (None, 137000))
def test_no_pulse_is_canonical_and_keeps_models_and_rng(profile, noise, monkeypatch):
    torch.set_num_threads(1)
    env, _ = worker.backend.physics_worker.make_env("id")
    model = worker.backend.make_model(env, 3, "cpu")
    factory = None if profile == "id" else worker.training.make_wrapped_env
    cell = dict(condition="id" if profile == "id" else "static/"+profile,
        mode="deterministic" if noise is None else "sampled", reset_seed=154000, noise_seed=noise)
    before, rng = deepcopy(model.policy.state_dict()), torch.get_rng_state()
    canonical = worker.backend.evaluate_episode(model, profile, cell, 256, env_factory=factory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("CPU evaluation must not enumerate CUDA"))
    observed = worker.evaluate_episode(model, profile, cell, worker.schedules()[0], 256)
    for key in ("dt", "prefix_markers", "recovery"): observed.pop(key)
    assert observed == canonical
    torch.testing.assert_close(before, model.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    model.env.close()


@pytest.mark.parametrize("profile,base", (("id", 1.), ("gravity-1p5", 1.5)))
def test_exact_pulse_length_and_switch_has_no_state_reset(profile, base):
    env = worker.PulseGravity(profile, dict(name="test", onset=2, duration=3))
    env.reset(seed=154000)
    raw = env.unwrapped
    original_mass, friction = raw.model.body_mass.copy(), raw.model.geom_friction.copy()
    gravity = env._base["gravity"].copy()
    observed = []
    for step in range(6):
        before = raw.data.qpos.copy(), raw.data.qvel.copy(), env.env._elapsed_steps
        env._maybe_switch_before_step()
        np.testing.assert_array_equal(raw.data.qpos, before[0]); np.testing.assert_array_equal(raw.data.qvel, before[1])
        assert env.env._elapsed_steps == before[2]
        _, _, term, trunc, info = env.step(np.zeros(3))
        assert not term and not trunc
        observed.append(info["ood_profile"])
        factor = worker.PEAK if 2 <= step < 5 else base
        np.testing.assert_array_equal(raw.model.opt.gravity, gravity*factor)
        np.testing.assert_array_equal(raw.model.body_mass, original_mass); np.testing.assert_array_equal(raw.model.geom_friction, friction)
    assert observed == [profile, profile, "gravity-only", "gravity-only", "gravity-only", profile]
    env.reset(seed=154000)
    np.testing.assert_array_equal(raw.model.opt.gravity, gravity*base)
    env.close()


def test_recovery_requires_health_at_the_end_of_the_full_horizon():
    schedule = dict(onset=100, duration=10)
    e = dict(profile_counts={"gravity-only": 10}, profile_events=[], length=210, physical_termination=True)
    assert worker.recovery(e, schedule)["recovery_evaluable"] and not worker.recovery(e, schedule)["recovered"]
    e["physical_termination"] = False
    assert worker.recovery(e, schedule)["recovered"]
    e["length"] = 150
    assert not worker.recovery(e, schedule)["recovery_evaluable"]
    e["profile_counts"] = {"id": 99}
    assert not worker.recovery(e, schedule)["exposed"]


def test_protocol_and_transport_copy_only_the_predeclared_final_zip(tmp_path, monkeypatch):
    assert len(worker.schedules()) == 9 and worker.CASES == transport.CASES
    assert worker.PEAK == 2.6591479484724942
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    row = dict(snapshots=[dict(step=300000, checkpoint="server/step300000.zip"), dict(step=600000, checkpoint="server/step600000.zip")], resumable_checkpoint="server/latest.pt")
    copies = []
    monkeypatch.setattr(transport.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(row)))
    monkeypatch.setattr(transport.transport, "copy_artifact", lambda p: copies.append(p) or tmp_path/"node/step600000.zip")
    path = transport.stage_source("seed3_id")
    assert copies == ["server/step600000.zip"]
    assert json.loads(path.read_text())["snapshots"] == row["snapshots"]


def test_task_logger_prevents_per_block_tmp_directory_creation(tmp_path, monkeypatch):
    env, _ = logging_backend.physics_worker.make_env("id")
    model = logging_backend.make_model(env, 3, "cpu")
    logging_backend.bind_task_logger(model, tmp_path)
    monkeypatch.setattr(__import__("stable_baselines3.common.utils", fromlist=["configure_logger"]), "configure_logger",
        lambda *a, **kw: pytest.fail("learn must reuse the task logger"))
    for _ in range(3): model.learn(4, reset_num_timesteps=False, log_interval=None)
    assert model.logger.dir == str(tmp_path/"sb3_log")
    model.env.close()
