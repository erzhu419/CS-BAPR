from dataclasses import asdict
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v140_critic_rate as worker
from scripts import run_general_ood_v140_multinode as transport


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_config_weights_and_protocol_change_only_critic_rate(profile):
    torch.set_num_threads(1)
    env, physics = worker.make_env(profile)
    torch.manual_seed(3)
    baseline = worker.v132.make_agent(env, "cpu", "mse-control")
    torch.manual_seed(3)
    probe = worker.make_agent(env, "cpu", "mse-control")
    expected = dict(asdict(baseline.config), lr_critic=worker.CRITIC_LR)
    assert asdict(probe.config) == expected
    assert all(g["lr"] == worker.CRITIC_LR for g in probe.critic_optimizer.param_groups)
    for name in ("actor", "critic", "target_critic", "actor_optimizer", "alpha_optimizer"):
        torch.testing.assert_close(getattr(probe, name).state_dict(), getattr(baseline, name).state_dict(), rtol=0., atol=0.)
    control = worker.v137 if profile == "id" else worker.v138
    expected_spec = dict(control.stage_spec(profile), study_revision=worker.REVISION,
                         lr_critic=worker.CRITIC_LR, control_lr_critic=3e-4, control_revision=control.REVISION)
    assert worker.stage_spec(profile) == expected_spec
    assert worker.evaluation_spec() == control.evaluation_spec()
    other, other_physics = control.make_env(profile)
    assert physics == other_physics
    state, _ = env.reset(seed=17)
    reference, _ = other.reset(seed=17)
    np.testing.assert_array_equal(state, reference)
    env.close(); other.close()


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_resume_retains_reduced_optimizer_rate_and_full_state(profile, tmp_path, monkeypatch):
    def pair(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env, _ = worker.make_env(profile)
        env._max_episode_steps = 3
        agent = worker.make_agent(env, "cpu", "mse-control")
        agent.config.batch_size = 8
        return env, agent
    spec = dict(worker.stage_spec(profile, True), steps=24, checkpoint_interval=4,
                diagnostic_interval=4, evaluation_interval=8, evaluation_max_steps=4)
    callback = lambda a, seed, episodes, limit: worker.v137.evaluate(a, profile, seed, episodes, limit, env_factory=worker.make_env)
    env, direct = pair(3)
    expected = worker.v132.train(direct, env, 3, spec, tmp_path/"direct/latest.pt", evaluation_fn=callback)
    env.close()
    save = worker.v132.save_progress
    def interrupt(*args):
        save(*args)
        if args[2]["step"] == 16:
            raise InterruptedError()
    monkeypatch.setattr(worker.v132, "save_progress", interrupt)
    env, partial = pair(3)
    checkpoint = tmp_path/"resume/latest.pt"
    with pytest.raises(InterruptedError):
        worker.v132.train(partial, env, 3, spec, checkpoint, evaluation_fn=callback)
    env.close()
    monkeypatch.setattr(worker.v132, "save_progress", save)
    env, resumed = pair(99)
    actual = worker.v132.train(resumed, env, 3, spec, checkpoint, checkpoint, evaluation_fn=callback)
    assert actual == expected and actual["optimizer_updates"] == 17
    for name in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        torch.testing.assert_close(getattr(direct, name).state_dict(), getattr(resumed, name).state_dict(), rtol=0., atol=0.)
    assert all(g["lr"] == worker.CRITIC_LR for g in resumed.critic_optimizer.param_groups)
    for lhs, rhs in zip(direct.replay_buffer.buffer, resumed.replay_buffer.buffer):
        for a, b in zip(lhs, rhs):
            np.testing.assert_array_equal(a, b)
    env.close()


def test_training_restore_use_same_factory_and_native_transport(tmp_path, monkeypatch):
    invoked = []
    monkeypatch.setattr(worker.v137, "run", lambda args, **kw: invoked.append(kw))
    args = SimpleNamespace(seed=3, profile="friction-only", checkpoint_dir=tmp_path/"ckpt",
                           result_dir=tmp_path/"result", resume_from=tmp_path/"ckpt/latest.pt", smoke=True)
    worker.run(args)
    assert invoked[0]["agent_factory"] is worker.make_agent
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    calls, published = [], []
    monkeypatch.setattr(transport.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    monkeypatch.setattr(transport.v132, "publish", lambda a: published.append(a))
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"} and published == [args]
    assert "--resume-from" in calls[0][0] and "--smoke" in calls[0][0]
    assert calls[0][0][calls[0][0].index("--device")+1] == "cuda"
