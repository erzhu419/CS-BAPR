from copy import deepcopy
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from csbapr.envs.extreme_ood import ExtremeOODConfig, ExtremeOODWrapper
from scripts import analyze_general_ood_v135_ood_context as worker
from scripts import run_general_ood_v135_multinode as transport
from test_general_ood_v134 import make_agent


def test_predeclared_four_task_budget_and_paired_cells():
    online, zero = (worker.stage_spec(3, mode) for mode in worker.CONTEXT_MODES)
    assert worker.cells(online) == worker.cells(zero)
    assert len(worker.cells(online)) == 120 and 4*120 == 480
    assert len(worker.cells(worker.stage_spec(3, "online", True))) == 6
    assert online["switch_every_steps"] == 50 and online["switch_probability"] == 1.
    assert online["reward_changes"] == [] and online["policy_receives_profile_labels"] is False


def test_static_physics_and_reward_match_declared_profile():
    env = worker.make_env("static/mega_event")
    state, info = env.reset(seed=152000)
    multipliers = worker.actual_multipliers(env)
    profile = env.current_profile
    assert info["ood_profile"] == "mega_event"
    assert multipliers["body_mass"] == pytest.approx(50.**.33)
    assert multipliers["gravity"] == pytest.approx(50.**.25)
    action = np.zeros(3, dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    assert reward == pytest.approx(1.+info["x_velocity"])
    assert profile["obs_scale"] == profile["action_scale"] == 1.
    env.close()


def test_markov_tick51_and101_repeat_without_policy_dependence():
    def trajectory():
        # Schedule test keeps stepping after unhealthy states, without training a policy.
        env = ExtremeOODWrapper(gym.make("Hopper-v4", terminate_when_unhealthy=False),
            config=ExtremeOODConfig(intensity=50., profile_family="structured", schedule="markov",
                switch_every_steps=50, switch_probability=1., include_sensor_noise=False,
                include_action_noise=False, include_reward_noise=False))
        env.reset(seed=152000)
        names = []
        for _ in range(101):
            _, _, _, _, info = env.step(np.zeros(3))
            names.append(info["ood_profile"])
            worker.actual_multipliers(env)
        env.close()
        return names
    names = trajectory()
    assert names == trajectory()
    assert names[:50] == ["id"]*50
    assert names[50] != "id" and len(set(names[50:100])) == 1
    assert names[100] != names[99]


@pytest.mark.parametrize("mode", worker.CONTEXT_MODES)
def test_context_input_ablation_is_frozen_and_preserves_normalizer(mode, monkeypatch):
    agent = make_agent()
    context = deepcopy(vars(agent.context_tracker))
    weights = deepcopy(agent.actor.state_dict())
    rng = torch.get_rng_state().clone()
    inputs = []
    handle = agent.actor.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    def unexpected_cuda_probe():
        raise AssertionError("CPU inference must not enumerate CUDA devices")
    monkeypatch.setattr(torch.cuda, "device_count", unexpected_cuda_probe)
    cell = dict(condition="id", mode="sampled", reset_seed=152000, noise_seed=135000)
    result = worker.evaluate_episode(agent, cell, mode, 16)
    handle.remove()
    assert result == worker.evaluate_episode(agent, cell, mode, 16)
    assert result["context_updates"] == (result["length"] if mode == "online" else 0)
    context_inputs = torch.cat(inputs)[:, agent.state_dim:]
    assert context_inputs.shape[1] == agent.context_dim
    assert bool(context_inputs.abs().max() > 0) == (mode == "online")
    assert result["native_reward_verified"] and result["non_id_steps"] == 0
    assert result["switches"] == 0 and result["profile_events"][0]["profile"] == "id"
    for key, value in context.items():
        np.testing.assert_equal(value, vars(agent.context_tracker)[key])
    torch.testing.assert_close(weights, agent.actor.state_dict(), rtol=0., atol=0.)
    assert torch.equal(rng, torch.get_rng_state())
    assert agent.training_steps == agent.replay_buffer.size == 0


def test_id_online_matches_existing_native_common_noise_evaluator():
    agent = make_agent()
    cell = dict(condition="id", mode="sampled", reset_seed=152000, noise_seed=135000)
    expected = worker.v134.evaluation.evaluate_episode(agent, 152000, 1., 135000, 16)
    result = worker.evaluate_episode(agent, cell, "online", 16)
    for key in ("length", "episode_return", "mean_velocity", "mean_policy_std", "mean_action_noise_l2"):
        assert result[key] == pytest.approx(expected[key])


def test_episode_resume_keeps_paired_schedule(tmp_path):
    agent = make_agent()
    agent.training_steps = 299745
    spec = dict(worker.stage_spec(3, "online", True), max_steps=8)
    direct = dict(evaluations=[])
    worker.collect(agent, spec, direct, lambda: None)
    partial, path = dict(evaluations=[]), tmp_path/"progress.json"
    def save():
        path.write_text(json.dumps(partial))
        if len(partial["evaluations"]) == 3:
            raise InterruptedError()
    with pytest.raises(InterruptedError):
        worker.collect(agent, spec, partial, save)
    resumed = json.loads(path.read_text())
    worker.collect(agent, spec, resumed, lambda: None)
    assert resumed == direct


def test_context_jobs_have_distinct_source_metadata_and_only_copy_final(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    source = dict(revision="general_ood_v133_source_extension", training_seed=3, arm="mse-control", smoke=False,
                  checkpoint=str(tmp_path/"central/final.pt"), resumable_checkpoint=str(tmp_path/"central/latest.pt"))
    monkeypatch.setattr(transport.subprocess, "run", lambda command, **kw: SimpleNamespace(stdout=json.dumps(source)))
    copies = []
    def copy(path):
        copies.append(path)
        return tmp_path/"node/final.pt"
    monkeypatch.setattr(transport.transport, "copy_artifact", copy)
    paths = [transport.stage_source(3, mode) for mode in worker.CONTEXT_MODES]
    assert paths[0] != paths[1]
    assert copies == [source["checkpoint"]]*2
    assert all(json.loads(path.read_text())["checkpoint"] == str(tmp_path/"node/final.pt") for path in paths)


def test_native_spawn_resume_and_scalar_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    monkeypatch.setattr(transport, "stage_source", lambda seed, mode: tmp_path/"node/source.json")
    calls = []
    monkeypatch.setattr(transport.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    result = tmp_path/"node/results"
    result.mkdir(parents=True)
    (result/"ood_context.json").write_text(json.dumps(dict(source=dict(checkpoint=str(tmp_path/"node/final.pt")))))
    args = SimpleNamespace(seed=3, context_mode="zero-context", checkpoint_dir=tmp_path/"node/progress", result_dir=result,
                           resume_from=tmp_path/"node/progress/progress.json", smoke=True)
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"}
    assert "--resume-from" in calls[0][0] and "--context-mode" in calls[0][0]
    copies = [command for command, _ in calls if command[0] == "rsync"]
    assert len(copies) == 1 and copies[0][-2].endswith("ood_context.json")
    assert not list(result.glob("*.pt"))
