from copy import deepcopy
from dataclasses import asdict
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v137_fixed_profile as worker
from scripts import run_general_ood_v137_multinode as transport
from test_general_ood_v132 import make_pair as normal_pair


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
    return lambda agent, seed, episodes, limit: worker.evaluate(agent, profile, seed, episodes, limit)


def test_six_fresh_matched_budgets_and_matching_final_cells():
    specs = [worker.stage_spec(p) for p in worker.PROFILES]
    assert sum(s["steps"] for s in specs)*2 == 1800000
    assert all(s["critic_td_objective"] == "mse" and s["bootstrap_mask"] == "terminated-only" for s in specs)
    assert all(s["intensity"] == 50. and s["effective_beta"] == -2. for s in specs)
    assert worker.stage_spec("mega_event", True)["steps"] == 512
    assert len(worker.evaluation_spec()["reset_seeds"])*(1+len(worker.evaluation_spec()["action_noise_seeds"]))*6 == 120
    agents = []
    for profile in worker.PROFILES:
        env, agent, _ = make_pair(profile)
        agents.append(agent)
        env.close()
    assert all(asdict(a.config) == asdict(agents[0].config) for a in agents)
    for a in agents[1:]:
        torch.testing.assert_close(a.actor.state_dict(), agents[0].actor.state_dict(), rtol=0., atol=0.)


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_physical_profile_persists_across_native_resets(profile):
    env, metadata = worker.make_env(profile)
    wrapper = worker.v135.make_env("id" if profile == "id" else "static/"+profile)
    for seed in (0, 17, 19):
        state, _ = env.reset(seed=seed)
        expected, _ = wrapper.reset(seed=seed)
        worker.assert_physics(env, metadata)
        np.testing.assert_array_equal(state.astype(np.float32), expected)
        np.testing.assert_array_equal(env.unwrapped.data.qpos, wrapper.unwrapped.data.qpos)
        np.testing.assert_array_equal(env.unwrapped.data.qvel, wrapper.unwrapped.data.qvel)
        for _ in range(3):
            action = np.array([.1, -.2, .3], dtype=np.float32)
            actual, reference = env.step(action), wrapper.step(action)
            np.testing.assert_array_equal(actual[0].astype(np.float32), reference[0])
            np.testing.assert_array_equal(env.unwrapped.data.qpos, wrapper.unwrapped.data.qpos)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, wrapper.unwrapped.data.qvel)
            assert actual[1:4] == reference[1:4]
            assert actual[4]["x_velocity"] == reference[4]["x_velocity"]
    env.close(); wrapper.close()


def test_id_callback_retains_original_driver_training_exactly(tmp_path):
    spec = dict(worker.stage_spec("id", True), steps=24, checkpoint_interval=4,
        diagnostic_interval=4, evaluation_interval=8, evaluation_max_steps=4)
    env, direct = normal_pair(3)
    expected = worker.v132.train(direct, env, 3, spec, tmp_path/"original/latest.pt")
    env.close()
    env, actual, _ = make_pair("id")
    result = worker.v132.train(actual, env, 3, spec, tmp_path/"callback/latest.pt", evaluation_fn=callback("id"))
    for evaluation in expected["periodic_evaluations"]:
        evaluation.update(profile="id", actual_multipliers=dict(body_mass=1., gravity=1., friction=1.))
    assert result == expected
    for name in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        torch.testing.assert_close(getattr(actual, name).state_dict(), getattr(direct, name).state_dict(), rtol=0., atol=0.)
    env.close()


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_full_state_resume_retains_profile_masks_and_weights(profile, tmp_path, monkeypatch):
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
    actual = worker.v132.train(resumed, env, 3, spec, checkpoint, checkpoint, evaluation_fn=callback(profile))
    assert actual == expected and actual["optimizer_updates"] == 17
    worker.assert_physics(env, physics)
    for name in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        torch.testing.assert_close(getattr(direct, name).state_dict(), getattr(resumed, name).state_dict(), rtol=0., atol=0.)
    for left, right in zip(direct.replay_buffer.buffer, resumed.replay_buffer.buffer):
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)
    assert sum(r[4] for r in resumed.replay_buffer.buffer) == sum(b["physical_terminations"] for b in actual["diagnostic_blocks"])
    assert any(b["time_limit_truncations"] for b in actual["diagnostic_blocks"])
    with pytest.raises(AssertionError):
        worker.v132.train(resumed, env, 3, dict(spec, training_profile="wrong"), checkpoint, checkpoint, evaluation_fn=callback(profile))
    env.close()


@pytest.mark.parametrize("profile", worker.PROFILES)
def test_periodic_evaluation_has_matching_physics_and_preserves_context(profile, monkeypatch):
    env, agent, physics = make_pair(profile)
    state, _ = env.reset(seed=3)
    agent.observe_context(state, np.zeros(3), 1., state+.1)
    before, weights = deepcopy(vars(agent.context_tracker)), deepcopy(agent.actor.state_dict())
    rng = torch.get_rng_state().clone()
    requested = []
    factory = worker.make_env
    def observed(name):
        requested.append(name)
        return factory(name)
    monkeypatch.setattr(worker, "make_env", observed)
    result = worker.evaluate(agent, profile, 149000, 2, 8)
    assert requested == [profile] and len(result["episodes"]) == 2
    for key, value in before.items():
        np.testing.assert_equal(value, vars(agent.context_tracker)[key])
    torch.testing.assert_close(weights, agent.actor.state_dict(), rtol=0., atol=0.)
    assert torch.equal(rng, torch.get_rng_state())
    assert agent.training_steps == agent.replay_buffer.size == 0
    worker.assert_physics(env, physics)
    env.close()


def test_final_cpu_evaluation_is_matching_and_no_replay(tmp_path, monkeypatch):
    env, agent, _ = make_pair("mega_event")
    agent.context_tracker._count = 512
    agent.training_steps = 257
    for model in (agent.actor, agent.critic, agent.target_critic):
        model.eval().requires_grad_(False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("CPU final inference must not enumerate CUDA"))
    result = worker.final_evaluation(agent, "mega_event", True)
    assert result["device"] == "cpu" and result["optimizer_updates"] == 0
    assert [e["mode"] for e in result["episodes"]] == ["deterministic", "sampled"]
    assert all(e["condition"] == "static/mega_event" and e["non_id_steps"] == e["length"] for e in result["episodes"])
    assert agent.context_tracker._count == 512 and agent.training_steps == 257
    env.close()


def test_native_cuda_spawn_and_server_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    calls, published = [], []
    monkeypatch.setattr(transport.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setattr(transport.v132, "publish", lambda args: published.append(args))
    args = SimpleNamespace(seed=3, profile="mega_event", checkpoint_dir=tmp_path/"progress", result_dir=tmp_path/"results",
        resume_from=tmp_path/"progress/latest.pt", smoke=True)
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"}
    assert calls[0][0][calls[0][0].index("--device")+1] == "cuda"
    assert "--resume-from" in calls[0][0] and "--smoke" in calls[0][0] and published == [args]
