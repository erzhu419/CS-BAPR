from copy import deepcopy
from dataclasses import asdict
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import run_general_ood_v138_physics_components as worker
from scripts import run_general_ood_v138_multinode as transport


def make_pair(profile, seed=3):
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env, physics = worker.make_env(profile)
    env._max_episode_steps = 3
    config = worker.v132.v39.v38.make_config("Hopper-v4", "fixed-penalty", "cpu")
    config.hidden_dim, config.batch_size, config.weight_reg = 16, 8, 0.
    agent = worker.v132.SourceStabilityAgent(11, 3, config, "mse-control")
    agent.surprise_computer = worker.v132.v39.RelativeSurpriseComputer(ema_alpha=config.surprise_ema)
    return env, agent, physics


def callback(profile):
    return lambda a, seed, episodes, limit: worker.v137.evaluate(a, profile, seed, episodes, limit, env_factory=worker.make_env)


def test_budget_and_noise_protocol_match_controls():
    assert sum(worker.stage_spec(p)["steps"] for p in worker.PROFILES)*2 == 1800000
    assert worker.evaluation_spec() == worker.v137.evaluation_spec()
    agents = []
    for profile in worker.PROFILES:
        spec = worker.stage_spec(profile)
        assert spec["active_factor"] == worker.FACTORS[profile] and spec["source_profile"] == "mega_event"
        assert spec["physics_factors"][worker.FACTORS[profile]] != 1.
        assert sum(v != 1. for v in spec["physics_factors"].values()) == 1
        env, agent, _ = make_pair(profile)
        agents.append(agent)
        env.close()
    assert all(asdict(a.config) == asdict(agents[0].config) for a in agents)
    for a in agents[1:]:
        torch.testing.assert_close(a.actor.state_dict(), agents[0].actor.state_dict(), rtol=0., atol=0.)


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_exact_single_factor_and_unchanged_actuator_inertia(profile):
    env, metadata = worker.make_env(profile)
    base = gym.make("Hopper-v4")
    model, reference = env.unwrapped.model, base.unwrapped.model
    expected = worker.profile_settings(profile)
    for field, scale in (("body_mass", expected["body_mass"]), ("geom_friction", expected["friction"])):
        np.testing.assert_array_equal(getattr(model, field), getattr(reference, field)*scale)
    np.testing.assert_array_equal(model.opt.gravity, reference.opt.gravity*expected["gravity"])
    for field in ("body_inertia", "actuator_gear", "actuator_ctrlrange", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(model, field), getattr(reference, field))
    wrapped = worker.make_wrapped_env("static/"+profile)
    for seed in (0, 17, 19):
        state, _ = env.reset(seed=seed)
        other, _ = wrapped.reset(seed=seed)
        np.testing.assert_array_equal(state.astype(np.float32), other)
        worker.v137.assert_physics(env, metadata)
        for _ in range(3):
            action = np.array([.1, -.2, .3], dtype=np.float32)
            actual, expected_step = env.step(action), wrapped.step(action)
            np.testing.assert_array_equal(actual[0].astype(np.float32), expected_step[0])
            np.testing.assert_array_equal(env.unwrapped.data.qpos, wrapped.unwrapped.data.qpos)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, wrapped.unwrapped.data.qvel)
            assert actual[1:4] == expected_step[1:4]
    env.close(); base.close(); wrapped.close()


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_component_resume_keeps_all_training_state(profile, tmp_path, monkeypatch):
    spec = dict(worker.stage_spec(profile, True), steps=24, checkpoint_interval=4,
        diagnostic_interval=4, evaluation_interval=8, evaluation_max_steps=4)
    env, direct, _ = make_pair(profile)
    expected = worker.v132.train(direct, env, 3, spec, tmp_path/"direct/latest.pt", evaluation_fn=callback(profile))
    env.close()
    save = worker.v132.save_progress
    def interrupt(*args):
        save(*args)
        if args[2]["step"] == 16:
            raise InterruptedError()
    monkeypatch.setattr(worker.v132, "save_progress", interrupt)
    env, partial, _ = make_pair(profile)
    checkpoint = tmp_path/"resume/latest.pt"
    with pytest.raises(InterruptedError):
        worker.v132.train(partial, env, 3, spec, checkpoint, evaluation_fn=callback(profile))
    env.close()
    monkeypatch.setattr(worker.v132, "save_progress", save)
    env, resumed, physics = make_pair(profile, 99)
    result = worker.v132.train(resumed, env, 3, spec, checkpoint, checkpoint, evaluation_fn=callback(profile))
    assert result == expected and result["optimizer_updates"] == 17
    assert all(e["profile"] == profile for e in result["periodic_evaluations"])
    for name in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        torch.testing.assert_close(getattr(direct, name).state_dict(), getattr(resumed, name).state_dict(), rtol=0., atol=0.)
    for lhs, rhs in zip(direct.replay_buffer.buffer, resumed.replay_buffer.buffer):
        for a, b in zip(lhs, rhs):
            np.testing.assert_array_equal(a, b)
    assert sum(r[4] for r in resumed.replay_buffer.buffer) == sum(b["physical_terminations"] for b in result["diagnostic_blocks"])
    assert any(b["time_limit_truncations"] for b in result["diagnostic_blocks"])
    worker.v137.assert_physics(env, physics)
    env.close()


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_final_episode_factory_preserves_context_and_cpu_inference(profile, monkeypatch):
    env, agent, _ = make_pair(profile)
    state, _ = env.reset(seed=3)
    agent.observe_context(state, np.zeros(3), 1., state+.1)
    before, weights = deepcopy(vars(agent.context_tracker)), deepcopy(agent.actor.state_dict())
    agent.training_steps = 257
    agent.actor.eval().requires_grad_(False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("CPU inference must not enumerate CUDA"))
    result = worker.final_evaluation(agent, profile, True)
    assert result["protocol"] == worker.evaluation_spec(True) and result["device"] == "cpu"
    assert result["optimizer_updates"] == 0 and not result["replay_loaded"]
    assert all(e["condition"] == "static/"+profile and e["native_reward_verified"] for e in result["episodes"])
    assert all(e["profile_counts"] == {profile: e["length"]} for e in result["episodes"])
    for key, value in before.items():
        np.testing.assert_equal(value, vars(agent.context_tracker)[key])
    torch.testing.assert_close(weights, agent.actor.state_dict(), rtol=0., atol=0.)
    env.close()


def test_shared_runner_and_native_transport_receive_component_factories(tmp_path, monkeypatch):
    invoked = []
    monkeypatch.setattr(worker.v137, "run", lambda args, **kw: invoked.append((args, kw)))
    args = SimpleNamespace(seed=3, profile="mass-only", checkpoint_dir=tmp_path/"progress", result_dir=tmp_path/"results",
        resume_from=tmp_path/"progress/latest.pt", smoke=True)
    worker.run(args)
    assert invoked[0][1] == dict(revision=worker.REVISION, env_factory=worker.make_env,
                                spec_factory=worker.stage_spec, final_evaluator=worker.final_evaluation)
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    calls, published = [], []
    monkeypatch.setattr(transport.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setattr(transport.v132, "publish", lambda a: published.append(a))
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"}
    assert calls[0][0][calls[0][0].index("--profile")+1] == "mass-only"
    assert "--resume-from" in calls[0][0] and "--smoke" in calls[0][0] and published == [args]
