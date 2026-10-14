from copy import deepcopy
import io
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import run_general_ood_v144_fixed_profile_sac as worker
from scripts import run_general_ood_v144_multinode as transport


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_static_physics_native_observation_and_reward_parity(profile):
    native, metadata = worker.physics_worker.make_env(profile)
    wrapped = worker.v137.v135.make_env("id") if profile == "id" else worker.v138.make_wrapped_env("static/"+profile)
    a, _ = native.reset(seed=154000)
    b, _ = wrapped.reset(seed=154000)
    np.testing.assert_array_equal(a.astype(np.float32), b)
    for _ in range(5):
        a, ra, ta, xa, _ = native.step(np.zeros(3))
        b, rb, tb, xb, _ = wrapped.step(np.zeros(3))
        np.testing.assert_array_equal(a.astype(np.float32), b)
        assert (ra, ta, xa) == (rb, tb, xb)
        worker.v137.assert_physics(native, metadata)
    native.close(); wrapped.close()


def test_standard_settings_noise_and_evaluation_do_not_change_model_or_rng(tmp_path):
    torch.set_num_threads(1)
    env, _ = worker.physics_worker.make_env("id")
    model = worker.make_model(env, 3, "cpu")
    model.learn(8, log_interval=None)
    assert model.observation_space.shape == (11,) and model.critic.n_critics == 2
    assert model.target_entropy == -3. and model.learning_starts == 255
    assert model.actor.optimizer.param_groups[0]["lr"] == .0003
    weights, rng = deepcopy(model.policy.state_dict()), torch.get_rng_state()
    numpy_rng = np.random.get_state()
    checkpoint = tmp_path/"model.zip"
    model.save(checkpoint)
    result = worker.final_evaluation(checkpoint, "id", True, "cpu")
    assert len(result["episodes"]) == 2 and result["optimizer_updates"] == 0 and not result["replay_loaded"]
    assert all(e["context_updates"] == 0 and e["native_reward_verified"] for e in result["episodes"])
    torch.testing.assert_close(weights, model.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    np.testing.assert_equal(numpy_rng, np.random.get_state())
    obs, _ = env.reset(seed=154000)
    expected, _ = model.predict(obs, deterministic=True)
    with torch.no_grad():
        mean, log_std, _ = model.actor.get_action_dist_params(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    np.testing.assert_array_equal(model.policy.unscale_action(mean.tanh().squeeze(0).numpy()), expected)
    assert torch.isfinite(log_std).all()
    model.env.close()


@pytest.mark.parametrize("failure", (False, True))
def test_sb3_replay_uses_terminated_only_mask(failure):
    env, _ = worker.physics_worker.make_env("id")
    if not failure:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1)
    model = worker.make_model(env, 3, "cpu")
    model._setup_learn(1, reset_num_timesteps=True)
    if failure:
        env.unwrapped.data.qpos[1] = .5
        __import__("mujoco").mj_forward(env.unwrapped.model, env.unwrapped.data)
    model.learn(1, reset_num_timesteps=False, log_interval=None)
    assert model.replay_buffer.dones[0, 0] == 1.
    assert model.replay_buffer.timeouts[0, 0] == float(not failure)
    sample = model.replay_buffer.sample(1)
    assert sample.dones.item() == float(failure)
    model.env.close()


def test_full_resume_matches_uninterrupted_native_sb3_training(tmp_path):
    torch.set_num_threads(1)
    env, _ = worker.physics_worker.make_env("friction-only")
    model = worker.make_model(env, 3, "cpu")
    model.learn(256, log_interval=None)
    spec = worker.stage_spec("friction-only", True)
    checkpoint = tmp_path/"latest.pt"
    worker.save_progress(model, env, dict(step=256), spec, 3, checkpoint)
    model.learn(256, reset_num_timesteps=False, log_interval=None)
    expected_weights, expected_rng = deepcopy(model.policy.state_dict()), torch.get_rng_state()
    expected_state = worker.environment_state(env)
    restored_env, _ = worker.physics_worker.make_env("friction-only")
    resumed, progress = worker.load_progress(checkpoint, restored_env, spec, 3, "cpu")
    assert progress["step"] == 256 and resumed.replay_buffer.size() == 256
    resumed.learn(256, reset_num_timesteps=False, log_interval=None)
    assert model.num_timesteps == resumed.num_timesteps == 512
    assert model._n_updates == resumed._n_updates == 257
    torch.testing.assert_close(expected_weights, resumed.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(expected_rng, torch.get_rng_state(), rtol=0., atol=0.)
    for key in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        np.testing.assert_array_equal(getattr(model.replay_buffer, key)[:512], getattr(resumed.replay_buffer, key)[:512])
    np.testing.assert_equal(expected_state, worker.environment_state(restored_env))
    model.env.close(); resumed.env.close()


def test_profile_reset_schedule_and_budget_milestones():
    env, _ = worker.physics_worker.make_env("gravity-only")
    seeded = worker.TrainingResets(env, 3)
    for seed in (30000, 30001):
        observed, _ = seeded.reset()
        expected, _ = env.reset(seed=seed)
        np.testing.assert_array_equal(observed, expected)
    seeded.close()
    spec = worker.stage_spec("id")
    assert spec["steps"] == 600000 and spec["milestones"] == [300000, 600000]
    assert spec["steps"]-worker.SETTINGS["learning_starts"] == 599745


def test_native_transport_publication_stays_server_side(tmp_path, monkeypatch):
    calls = []
    node, central = tmp_path/"node", tmp_path/"central"
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", node)
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", central)
    monkeypatch.setattr(transport.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    result = node/"run_results/seed3_id"
    result.mkdir(parents=True)
    snapshots = [dict(step=n, checkpoint=str(node/f"run_checkpoints/seed3_id/step{n}.zip")) for n in (300000, 600000)]
    (result/"source.json").write_text(json.dumps(dict(resumable_checkpoint=str(node/"run_checkpoints/seed3_id/latest.pt"), snapshots=snapshots, training=dict(snapshots=snapshots))))
    transport.publish(SimpleNamespace(result_dir=result))
    assert all(str(central) in " ".join(cmd) for cmd, kw in calls if cmd[0] == "rsync")
    row = json.loads((result/"source.json").read_text())
    assert row["snapshots"][0]["checkpoint"] == str(central/"run_checkpoints/seed3_id/step300000.zip")
    assert row["training"]["snapshots"] == row["snapshots"]
