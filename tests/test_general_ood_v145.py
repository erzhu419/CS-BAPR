from copy import deepcopy
import io

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v145_gravity_calibration as worker


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_only_gravity_changes_and_native_reset_preserves_physics(profile):
    native, physics = worker.make_env(profile)
    base, original = worker.backend.physics_worker.make_env("id")
    np.testing.assert_array_equal(physics["body_mass"], original["body_mass"])
    np.testing.assert_array_equal(physics["geom_friction"], original["geom_friction"])
    np.testing.assert_allclose(physics["gravity"], np.array(original["gravity"])*worker.GRAVITY[profile], rtol=0., atol=0.)
    assert physics["actual_multipliers"] == dict(body_mass=1., gravity=worker.GRAVITY[profile], friction=1.)
    for seed in (30000, 30001):
        native.reset(seed=seed)
        worker.backend.v137.assert_physics(native, physics)
    wrapped = worker.make_wrapped_env(profile)
    a, _ = native.reset(seed=154000)
    b, _ = wrapped.reset(seed=154000)
    np.testing.assert_array_equal(a.astype(np.float32), b)
    for _ in range(5):
        a, ra, ta, xa, _ = native.step(np.zeros(3))
        b, rb, tb, xb, _ = wrapped.step(np.zeros(3))
        np.testing.assert_array_equal(a.astype(np.float32), b)
        assert (ra, ta, xa) == (rb, tb, xb)
    assert native.unwrapped._terminate_when_unhealthy and native.unwrapped._healthy_z_range[0] == .7
    assert native.unwrapped._healthy_reward == base.unwrapped._healthy_reward == 1.
    native.close(); base.close(); wrapped.close()


def test_predeclared_budgets_and_original_peak_are_retained():
    for profile in worker.PROFILES:
        spec = worker.stage_spec(profile)
        assert spec["steps"] == 600000 and spec["milestones"] == [300000, 600000]
        assert spec["heldout_peak_gravity_multiplier"] == 2.6591479484724942
        assert spec["settings"] == worker.backend.SETTINGS


def test_both_cpu_evaluations_preserve_weights_and_rng(tmp_path):
    torch.set_num_threads(1)
    env, _ = worker.make_env("gravity-1p5")
    model = worker.backend.make_model(env, 3, "cpu")
    checkpoint = tmp_path/"policy.zip"
    model.save(checkpoint)
    before, rng = deepcopy(model.policy.state_dict()), torch.get_rng_state()
    result = worker.final_evaluation(checkpoint, "gravity-1p5", True, "cpu")
    assert result["profile"] == "gravity-1p5" and result["heldout_peak_evaluation"]["profile"] == "gravity-only"
    for final, factor in ((result, 1.5), (result["heldout_peak_evaluation"], 2.6591479484724942)):
        assert final["device"] == "cpu" and final["optimizer_updates"] == 0 and not final["replay_loaded"]
        assert len(final["episodes"]) == 2
        assert all(e["profile_events"][0]["actual_multipliers"]["gravity"] == factor for e in final["episodes"])
    torch.testing.assert_close(before, model.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    model.env.close()


def test_calibration_revision_survives_exact_full_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(worker.backend.SETTINGS, "buffer_size", 2048)
    env, _ = worker.make_env("gravity-2p0")
    model = worker.backend.make_model(env, 4, "cpu")
    model.learn(256, log_interval=None)
    spec = worker.stage_spec("gravity-2p0", True)
    path = tmp_path/"latest.pt"
    worker.backend.save_progress(model, env, dict(step=256), spec, 4, path)
    model.learn(256, reset_num_timesteps=False, log_interval=None)
    expected, rng, physics = deepcopy(model.policy.state_dict()), torch.get_rng_state(), worker.backend.environment_state(env)
    restored_env, _ = worker.make_env("gravity-2p0")
    restored, _ = worker.backend.load_progress(path, restored_env, spec, 4, "cpu")
    restored.learn(256, reset_num_timesteps=False, log_interval=None)
    assert restored._n_updates == 257
    torch.testing.assert_close(expected, restored.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    np.testing.assert_equal(physics, worker.backend.environment_state(restored_env))
    model.env.close(); restored.env.close()
