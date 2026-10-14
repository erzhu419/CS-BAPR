from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v134_frozen_robustness as worker
from scripts import run_general_ood_v134_multinode as transport


def make_agent():
    torch.set_num_threads(1)
    env = gym.make("Hopper-v4")
    agent = worker.v133.v132.make_agent(env, "cpu", "mse-control")
    state, _ = env.reset(seed=134)
    agent.observe_context(state, np.zeros(3), 1., state+.1)
    agent.observe_context(state+.1, np.zeros(3), 2., state+.2)
    env.close()
    agent.actor.eval().requires_grad_(False)
    return agent


def test_protocol_covers_all_anchors_resets_and_common_noise():
    spec = worker.stage_spec(3, "mse-control")
    cells = worker.cells(spec)
    assert len(cells) == 85 and 4*len(cells) == 340
    assert len({(c["group"], c["mode"], c["reset_seed"], c["noise_seed"]) for c in cells}) == 85
    assert sum(c["group"] == "anchor" for c in cells) == 5
    assert sum(c["mode"] == "sampled" for c in cells) == 60
    assert set(spec["anchor_reset_seeds"]).isdisjoint(spec["new_reset_seeds"])
    assert spec["sampled_noise_scale"] == 1. and spec["max_steps"] == 1000
    assert len(worker.cells(worker.stage_spec(4, "huber10", True))) == 3


def test_compact_episode_keeps_context_rng_weights_and_empty_replay(monkeypatch):
    agent = make_agent()
    before = deepcopy(vars(agent.context_tracker))
    weights = deepcopy(agent.actor.state_dict())
    rng = torch.get_rng_state().clone()
    def unexpected_cuda_probe():
        raise AssertionError("CPU-only evaluation must not enumerate CUDA devices")
    monkeypatch.setattr(torch.cuda, "device_count", unexpected_cuda_probe)
    cell = dict(group="new-reset", mode="sampled", reset_seed=151000, noise_seed=134000)
    result = worker.evaluate_cell(agent, cell, 16)
    assert result == worker.evaluate_cell(agent, cell, 16)
    assert result["mode"] == "sampled" and result["noise_scale"] == 1.
    assert "tail" not in result and "same_input_parity" not in result
    assert agent.replay_buffer.size == agent.training_steps == 0
    for key, value in before.items():
        np.testing.assert_equal(value, vars(agent.context_tracker)[key])
    torch.testing.assert_close(weights, agent.actor.state_dict(), rtol=0., atol=0.)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in agent.actor.parameters())
    assert not torch.cuda.is_initialized()


def test_episode_progress_resume_matches_complete_evaluation(tmp_path):
    agent = make_agent()
    spec = dict(worker.stage_spec(3, "mse-control", True), max_steps=8)
    expected = dict(evaluations=[])
    worker.collect(agent, spec, expected, lambda: None)
    partial = dict(evaluations=[])
    path = tmp_path/"progress.json"
    def save():
        path.write_text(json.dumps(partial))
        if len(partial["evaluations"]) == 2:
            raise InterruptedError()
    with pytest.raises(InterruptedError):
        worker.collect(agent, spec, partial, save)
    resumed = json.loads(path.read_text())
    worker.collect(agent, spec, resumed, lambda: None)
    assert resumed == expected
    resumed["evaluations"][0]["reset_seed"] += 1
    with pytest.raises(AssertionError):
        worker.collect(agent, spec, resumed, lambda: None)


def test_loading_lightweight_final_preserves_trained_context_without_replay(tmp_path):
    agent = make_agent()
    agent.training_steps, agent.context_tracker._count = 299745, 300000
    path = tmp_path/"final.pt"
    agent.save(str(path))
    context = deepcopy(vars(agent.context_tracker))
    versions = dict(torch=torch.__version__)
    source = dict(revision=worker.v133.REVISION, training_seed=3, arm="mse-control", smoke=False,
        spec=worker.v133.stage_spec("mse-control"), training=dict(interactions=300000, optimizer_updates=299745),
        checkpoint_selection="final-not-best", source_readout_refitted=False, periodic_evaluation_selects_checkpoint=False,
        runtime=versions, config=asdict(agent.config), checkpoint=str(path))
    loaded = worker.load_source(source, 3, "mse-control", versions)
    assert loaded.replay_buffer.size == 0 and loaded.training_steps == 299745
    for key, value in context.items():
        np.testing.assert_equal(value, vars(loaded.context_tracker)[key])
    assert all(not p.requires_grad for model in (loaded.actor, loaded.critic, loaded.target_critic) for p in model.parameters())
    with pytest.raises(AssertionError):
        worker.load_source(source, 3, "huber10", versions)


def test_server_stage_fetches_final_actor_only(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    row = dict(revision=worker.v133.REVISION, training_seed=4, arm="huber10", smoke=False,
               checkpoint=str(tmp_path/"central/final.pt"), resumable_checkpoint=str(tmp_path/"central/latest.pt"))
    monkeypatch.setattr(transport.subprocess, "run", lambda command, **kw: SimpleNamespace(stdout=json.dumps(row)))
    copies = []
    def copy(path):
        copies.append(path)
        return tmp_path/"node/final.pt"
    monkeypatch.setattr(transport.transport, "copy_artifact", copy)
    result = transport.stage_source(4, "huber10")
    assert copies == [row["checkpoint"]]
    assert json.loads(result.read_text())["checkpoint"] == str(tmp_path/"node/final.pt")
    with pytest.raises(AssertionError):
        transport.stage_source(4, "mse-control")
    assert len(copies) == 1


def test_native_cpu_spawn_resume_and_scalar_only_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    monkeypatch.setattr(transport, "stage_source", lambda seed, arm: tmp_path/"node/source.json")
    calls = []
    monkeypatch.setattr(transport.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    result = tmp_path/"node/results"
    result.mkdir(parents=True)
    (result/"robustness.json").write_text(json.dumps(dict(source=dict(checkpoint=str(tmp_path/"node/final.pt")))))
    args = SimpleNamespace(seed=4, arm="huber10", checkpoint_dir=tmp_path/"node/checkpoints", result_dir=result,
                           resume_from=tmp_path/"node/checkpoints/progress.json", smoke=True)
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"}
    assert "--resume-from" in calls[0][0] and "--smoke" in calls[0][0]
    copies = [args for args, _ in calls if args[0] == "rsync"]
    assert len(copies) == 1 and copies[0][-2].endswith("robustness.json")
    assert not list(result.glob("*.pt"))
    assert json.loads((result/"robustness.json").read_text())["source"]["checkpoint"] == str(tmp_path/"central/final.pt")
