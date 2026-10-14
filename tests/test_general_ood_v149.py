from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v149_warm_recovery as worker
from scripts import run_general_ood_v149_multinode as transport


def source_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(worker.backend.SETTINGS, "buffer_size", 2048)
    env = worker.training.RandomPulseGravity("no-pulse", 5)
    model = worker.backend.make_model(env, 5, "cpu")
    worker.backend.bind_task_logger(model, tmp_path/"original")
    model.learn(256, log_interval=None)
    spec = worker.training.stage_spec("no-pulse", True)
    path = tmp_path/"source.pt"
    worker.training.save_progress(model, env, dict(step=256), spec, 5, path)
    return model, env, spec, path


@pytest.mark.parametrize("arm", worker.ARMS)
def test_warm_transfer_preserves_weights_optimizers_replay_rng_and_current_episode(tmp_path, monkeypatch, arm):
    model, env, spec, path = source_checkpoint(tmp_path, monkeypatch)
    weights = deepcopy(model.policy.state_dict())
    optimizers = [deepcopy(o.state_dict()) for o in (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer)]
    entropy, state = model.log_ent_coef.detach().clone(), worker.training.environment_state(env)
    torch_rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
    restored_env = worker.training.RandomPulseGravity(arm, 5)
    restored, receipt = worker.warm_start(path, restored_env, spec, 5, "cpu")
    assert receipt["step"] == receipt["replay_size"] == receipt["replay_position"] == 256
    assert receipt["optimizer_steps"] == dict(actor=[1], critic=[1], entropy=[1])
    torch.testing.assert_close(weights, restored.policy.state_dict(), rtol=0., atol=0.)
    for expected, actual in zip(optimizers, (restored.actor.optimizer, restored.critic.optimizer, restored.ent_coef_optimizer)):
        torch.testing.assert_close(expected, actual.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(entropy, restored.log_ent_coef, rtol=0., atol=0.)
    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0., atol=0.)
    np.testing.assert_equal(numpy_rng, np.random.get_state())
    np.testing.assert_equal(state, worker.training.environment_state(restored_env))
    for key in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        np.testing.assert_array_equal(getattr(model.replay_buffer, key)[:256], getattr(restored.replay_buffer, key)[:256])
    assert restored_env.onset is None and restored_env.arm == arm
    expected_rng = np.random.RandomState(147005)
    selected = arm == "randomized-recovery" and expected_rng.rand() < .5
    onset = int(expected_rng.randint(80, 321)) if selected else None
    restored.env.reset()
    assert restored_env.onset == onset
    model.env.close(); restored.env.close()


def test_control_continuation_matches_uninterrupted_training(tmp_path, monkeypatch):
    model, env, source_spec, path = source_checkpoint(tmp_path, monkeypatch)
    model.learn(64, reset_num_timesteps=False, log_interval=None)
    weights, rng, state = deepcopy(model.policy.state_dict()), torch.get_rng_state(), worker.training.environment_state(env)
    restored_env = worker.training.RandomPulseGravity("no-pulse", 5)
    restored, _ = worker.warm_start(path, restored_env, source_spec, 5, "cpu")
    worker.backend.bind_task_logger(restored, tmp_path/"restored")
    restored.learn(64, reset_num_timesteps=False, log_interval=None)
    torch.testing.assert_close(weights, restored.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    np.testing.assert_equal(state, worker.training.environment_state(restored_env))
    assert restored.num_timesteps == 320 and restored._n_updates == 65
    model.env.close(); restored.env.close()


def test_target_revision_full_resume_preserves_pulse_clock_and_learning(tmp_path, monkeypatch):
    model, source_env, old_spec, path = source_checkpoint(tmp_path, monkeypatch)
    env = worker.training.RandomPulseGravity("randomized-recovery", 5)
    restored, _ = worker.warm_start(path, env, old_spec, 5, "cpu")
    worker.backend.bind_task_logger(restored, tmp_path/"first")
    restored._last_obs = restored.env.reset()
    restored._last_episode_starts = np.ones(1, dtype=bool)
    env.onset = 0
    restored.learn(3, reset_num_timesteps=False, log_interval=None)
    assert env.active_profile_name == "gravity-only"
    spec = worker.stage_spec("randomized-recovery", True)
    target = tmp_path/"latest.pt"
    worker.training.save_progress(restored, env, dict(step=259), spec, 5, target)
    assert torch.load(target, weights_only=False)["revision"] == worker.REVISION
    restored.learn(64, reset_num_timesteps=False, log_interval=None)
    weights, rng, state = deepcopy(restored.policy.state_dict()), torch.get_rng_state(), worker.training.environment_state(env)
    resumed_env = worker.training.RandomPulseGravity("randomized-recovery", 5)
    resumed, progress = worker.training.load_progress(target, resumed_env, spec, 5, "cpu")
    worker.backend.bind_task_logger(resumed, tmp_path/"resumed")
    assert progress["step"] == 259 and resumed_env.active_profile_name == "gravity-only"
    resumed.learn(64, reset_num_timesteps=False, log_interval=None)
    torch.testing.assert_close(weights, resumed.policy.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    np.testing.assert_equal(state, worker.training.environment_state(resumed_env))
    model.env.close(); restored.env.close(); resumed.env.close()


def test_new_budget_and_heldout_protocol_are_unchanged():
    for arm in worker.ARMS:
        spec = worker.stage_spec(arm)
        assert spec["steps"] == 700000 and spec["source_steps"] == 600000 and spec["new_steps"] == 100000
        assert spec["milestones"] == [650000, 700000]
        assert spec["training_pulse"]["gravity"] == 2. and spec["training_pulse"]["duration"] == 10
        assert spec["heldout_peak_gravity"] == 2.6591479484724942 and spec["heldout_durations"] == [25, 50]
        assert spec["settings"] == worker.backend.SETTINGS


def test_transport_reuses_read_only_native_source_and_resume_skips_full_transfer(tmp_path, monkeypatch):
    node, central = tmp_path/"node", tmp_path/"central"
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", node)
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", central)
    row = dict(resumable_checkpoint=str(central/"old/latest.pt"))
    native = node/"old/latest.pt"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"source")
    calls = []
    monkeypatch.setattr(transport.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(stdout=json.dumps(row)))
    path = transport.stage_source(5, "no-pulse", False)
    assert json.loads(path.read_text())["initialization_checkpoint"] == str(native)
    assert not any(c[0] == "rsync" for c in calls) and native.read_bytes() == b"source"
    other = transport.stage_source(5, "randomized-recovery", False)
    assert other != path
    native.unlink()
    path = transport.stage_source(5, "no-pulse", True)
    assert not any(c[0] == "rsync" for c in calls)
