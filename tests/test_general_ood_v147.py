from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v147_randomized_recovery as worker


@pytest.mark.parametrize("seed", (5, 6))
def test_no_pulse_control_is_native_and_keeps_private_rng(seed):
    env = worker.RandomPulseGravity("no-pulse", seed)
    native, _ = worker.backend.physics_worker.make_env("id")
    rng = env.pulse_rng.get_state()
    for reset in (seed*10000, seed*10000+1):
        a, _ = env.reset(seed=reset)
        b, _ = native.reset(seed=reset)
        np.testing.assert_array_equal(a, b.astype(np.float32))
        for _ in range(8):
            action = np.array([.1, -.2, .3], dtype=np.float32)
            a, ra, ta, xa, _ = env.step(action)
            b, rb, tb, xb, _ = native.step(action)
            np.testing.assert_array_equal(a, b.astype(np.float32))
            assert (ra, ta, xa) == (rb, tb, xb)
            worker.assert_physics(env)
    np.testing.assert_equal(rng, env.pulse_rng.get_state())
    assert env.stats["pulse_steps"] == env.stats["pulse_assigned_episodes"] == 0
    env.close(); native.close()


@pytest.mark.parametrize("seed", (5, 6))
def test_assignment_matches_private_rng_not_policy_or_reset_rng(seed):
    env = worker.RandomPulseGravity("randomized-recovery", seed)
    private = np.random.RandomState(147000+seed)
    expected = []
    before = np.random.get_state(), torch.get_rng_state()
    for reset in range(100):
        selected = private.rand() < .5
        expected.append(int(private.randint(80, 321)) if selected else None)
        env.reset(seed=seed*10000+reset)
        assert env.onset == expected[-1]
    np.testing.assert_equal(before[0], np.random.get_state())
    torch.testing.assert_close(before[1], torch.get_rng_state(), rtol=0., atol=0.)
    assert any(v is None for v in expected) and any(v is not None for v in expected)
    assert all(80 <= v <= 320 for v in expected if v is not None)
    env.close()


def test_exact_2x_ten_step_pulse_and_no_switch_reset():
    env = worker.RandomPulseGravity("randomized-recovery", 5)
    env.reset(seed=154000)
    env.onset = 2
    active = []
    for step in range(14):
        qpos, qvel, elapsed = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy(), env.env._elapsed_steps
        env._maybe_switch_before_step()
        np.testing.assert_array_equal(env.unwrapped.data.qpos, qpos)
        np.testing.assert_array_equal(env.unwrapped.data.qvel, qvel)
        assert env.env._elapsed_steps == elapsed
        _, _, term, trunc, info = env.step(np.zeros(3))
        assert not term and not trunc
        active.append(info["ood_profile"])
        worker.assert_physics(env)
    assert active == ["id"]*2+["gravity-only"]*10+["id"]*2
    assert env.stats["pulse_steps"] == 10 and env.stats["baseline_steps"] == 4
    assert env.stats["pulse_restored_episodes"] == 1
    env.reset(seed=154000)
    np.testing.assert_array_equal(env.unwrapped.model.opt.gravity, env._base["gravity"])
    env.close()


@pytest.mark.parametrize("arm", worker.ARMS)
def test_full_resume_during_pulse_matches_training_and_future_resets(tmp_path, monkeypatch, arm):
    torch.set_num_threads(1)
    monkeypatch.setitem(worker.backend.SETTINGS, "buffer_size", 2048)
    env = worker.RandomPulseGravity(arm, 5)
    model = worker.backend.make_model(env, 5, "cpu")
    worker.backend.bind_task_logger(model, tmp_path/"first")
    model.learn(256, log_interval=None)
    if arm == "randomized-recovery":
        model._last_obs = model.env.reset()
        model._last_episode_starts = np.ones(1, dtype=bool)
        env.onset = 0
        model.learn(3, reset_num_timesteps=False, log_interval=None)
        assert env.active_profile_name == "gravity-only"
    step = model.num_timesteps
    checkpoint = tmp_path/"latest.pt"
    spec = worker.stage_spec(arm, True)
    worker.save_progress(model, env, dict(step=step), spec, 5, checkpoint)
    model.learn(512-step, reset_num_timesteps=False, log_interval=None)
    weights, rng, state = deepcopy(model.policy.state_dict()), torch.get_rng_state(), worker.environment_state(env)
    resumed_env = worker.RandomPulseGravity(arm, 5)
    resumed, progress = worker.load_progress(checkpoint, resumed_env, spec, 5, "cpu")
    worker.backend.bind_task_logger(resumed, tmp_path/"resumed")
    assert progress["step"] == step
    if arm == "randomized-recovery":
        assert resumed_env.active_profile_name == "gravity-only"
    resumed.learn(512-step, reset_num_timesteps=False, log_interval=None)
    assert resumed.num_timesteps == model.num_timesteps == 512
    assert resumed._n_updates == model._n_updates == 257
    torch.testing.assert_close(weights, resumed.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    np.testing.assert_equal(state, worker.environment_state(resumed_env))
    for key in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        np.testing.assert_array_equal(getattr(model.replay_buffer, key)[:512], getattr(resumed.replay_buffer, key)[:512])
    model.env.close(); resumed.env.close()


def test_predeclared_budget_and_v146_evaluation_keep_rng_and_weights(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(worker.backend.SETTINGS, "buffer_size", 2048)
    for arm in worker.ARMS:
        spec = worker.stage_spec(arm)
        assert spec["steps"] == 600000 and spec["milestones"] == [300000, 600000]
        assert spec["heldout_peak_gravity"] == 2.6591479484724942 and spec["heldout_durations"] == [25, 50]
        assert spec["settings"] == worker.backend.SETTINGS
    env = worker.RandomPulseGravity("randomized-recovery", 6)
    model = worker.backend.make_model(env, 6, "cpu")
    checkpoint = tmp_path/"policy.zip"
    model.save(checkpoint)
    weights, rng = deepcopy(model.policy.state_dict()), torch.get_rng_state()
    result = worker.final_evaluation(checkpoint, True, "cpu")
    assert len(result["episodes"]) == 6 and result["prefix_pairs_verified"] == 4
    assert result["optimizer_updates"] == 0 and not result["replay_loaded"]
    assert result["protocol"]["schedules"] == worker.evaluation.schedules(True)
    torch.testing.assert_close(weights, model.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    model.env.close()
