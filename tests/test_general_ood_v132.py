from copy import deepcopy
from dataclasses import asdict
import json
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from scripts import run_general_ood_v132_source_stability as worker
from scripts import run_general_ood_v132_multinode as transport


def make_pair(seed=3, arm="mse-control"):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("Hopper-v4", max_episode_steps=3)
    config = worker.v39.v38.make_config("Hopper-v4", "fixed-penalty", "cpu")
    config.hidden_dim, config.batch_size, config.weight_reg = 16, 8, 0.
    agent = worker.SourceStabilityAgent(11, 3, config, arm)
    agent.surprise_computer = worker.v39.RelativeSurpriseComputer(ema_alpha=config.surprise_ema)
    return env, agent


def test_budget_and_matched_configs():
    assert worker.SEEDS == (3, 4)
    assert sum(worker.stage_spec(a)["steps"] for a in worker.ARMS)*2 == 400000
    env = gym.make("Hopper-v4")
    agents = [worker.make_agent(env, "cpu", arm) for arm in worker.ARMS]
    assert asdict(agents[0].config) == asdict(agents[1].config)
    assert agents[0].config.grad_clip == 1. and agents[0].config.penalty_scale == 0.
    assert env.spec.max_episode_steps == 1000
    env.close()


@pytest.mark.parametrize("terminated,truncated,mask", [(False, True, 0.), (True, False, 1.), (True, True, 1.)])
def test_transition_separates_bootstrap_from_context_boundary(terminated, truncated, mask):
    env, agent = make_pair()
    state, _ = env.reset(seed=3)
    expected = deepcopy(agent.context_tracker)
    action, next_state = np.zeros(3), state+.1
    next_context = expected.update(state, action, 2., next_state, done=terminated or truncated)
    agent.store_transition(state, action, 2., next_state, terminated, truncated)
    row = agent.replay_buffer.buffer[-1]
    assert row[4] == mask
    np.testing.assert_array_equal(row[5], np.zeros(agent.context_dim))
    np.testing.assert_array_equal(row[6], next_context)
    assert agent.context_tracker._count == 1
    agent.reset_context()
    assert agent.context_tracker._count == 1
    np.testing.assert_array_equal(agent._current_context_np(), np.zeros(agent.context_dim))
    env.close()


def test_huber_matches_mse_inliers_and_caps_outlier_derivative():
    residual = torch.tensor([-100., -10., -3., 0., 3., 10., 100.], requires_grad=True)
    target = torch.zeros_like(residual)
    mse = worker.td_objective(residual, target, "mse-control")
    huber = worker.td_objective(residual, target, "huber10")
    actual = torch.autograd.grad(huber, residual)[0]*residual.numel()
    torch.testing.assert_close(actual, 2.*residual.detach().clamp(-10., 10.))
    assert huber < mse
    x = residual.detach()[1:-1].requires_grad_()
    quadratic, robust = (worker.td_objective(x, torch.zeros_like(x), arm) for arm in worker.ARMS)
    torch.testing.assert_close(quadratic, robust)
    torch.testing.assert_close(torch.autograd.grad(quadratic, x)[0], torch.autograd.grad(robust, x)[0])


def test_mse_control_exactly_matches_native_and_huber_keeps_disagreement():
    env, agent = make_pair()
    batch = 4
    args = (torch.randn(batch, 11+agent.context_dim), torch.randn(batch, 3), torch.randn(batch, 1),
            torch.randn(batch, 11+agent.context_dim), torch.tensor([[0.], [1.], [0.], [1.]]),
            torch.randn(batch, 3), torch.randn(batch), torch.ones(agent.critic.num_critics), batch)
    native = CSBAPRAgent._compute_q_loss_bapr_fallback(agent, *args)
    corrected = agent._compute_q_loss_bapr_fallback(*args)
    for lhs, rhs in zip(native, corrected):
        torch.testing.assert_close(lhs, rhs, rtol=0., atol=0.)
    agent.arm = "huber10"
    robust = agent._compute_q_loss_bapr_fallback(*args)
    torch.testing.assert_close(robust[2], native[2])
    torch.testing.assert_close(robust[0], torch.tensor(agent.latest_td_metrics["td_objective"])+agent.config.beta_ood*robust[2])
    env.close()


@pytest.mark.parametrize("arm", worker.ARMS)
def test_exact_resume_preserves_episode_resets_and_new_masks(tmp_path, monkeypatch, arm):
    spec = dict(worker.stage_spec(arm, True), steps=24, checkpoint_interval=4, diagnostic_interval=4,
                evaluation_interval=8, evaluation_max_steps=4)
    env, direct = make_pair(3, arm)
    expected = worker.train(direct, env, 3, spec, tmp_path/"direct/latest.pt")
    env.close()
    assert any(r["time_limit_truncations"] for r in expected["diagnostic_blocks"])
    assert sum(row[4] for row in direct.replay_buffer.buffer) == sum(r["physical_terminations"] for r in expected["diagnostic_blocks"])
    save = worker.save_progress
    def interrupt(agent, env, loop, spec, seed, path):
        save(agent, env, loop, spec, seed, path)
        if loop["step"] == 16:
            raise InterruptedError()
    monkeypatch.setattr(worker, "save_progress", interrupt)
    env, partial = make_pair(3, arm)
    checkpoint = tmp_path/"resume/latest.pt"
    with pytest.raises(InterruptedError):
        worker.train(partial, env, 3, spec, checkpoint)
    env.close()
    monkeypatch.setattr(worker, "save_progress", save)
    env, resumed = make_pair(99, arm)
    actual = worker.train(resumed, env, 3, spec, checkpoint, checkpoint)
    assert actual == expected and actual["optimizer_updates"] == 17
    for name in ("actor", "critic", "target_critic"):
        for key, value in getattr(direct, name).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, name).state_dict()[key], rtol=0., atol=0.)
    wrong_arm = next(a for a in worker.ARMS if a != arm)
    with pytest.raises(AssertionError):
        worker.train(resumed, env, 3, dict(spec, arm=wrong_arm), checkpoint, checkpoint)
    env.close()


def test_new_node_native_spawn_and_server_only_publication(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    monkeypatch.setattr(transport.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    result = tmp_path/"node/results"
    result.mkdir(parents=True)
    checkpoint = tmp_path/"node/checkpoints"
    (result/"source.json").write_text(json.dumps(dict(checkpoint=str(checkpoint/"final.pt"),
        resumable_checkpoint=str(checkpoint/"latest.pt"))))
    args = SimpleNamespace(seed=3, arm="huber10", checkpoint_dir=checkpoint, result_dir=result,
                           resume_from=checkpoint/"latest.pt", smoke=True)
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"}
    assert "--resume-from" in calls[0][0] and "--smoke" in calls[0][0]
    copies = [args for args, _ in calls if args[0] == "rsync"]
    assert len(copies) == 3
    assert all(args[-1].startswith(transport.transport.PEER+":") for args in copies)
    row = json.loads((result/"source.json").read_text())
    assert row["resumable_checkpoint"] == str(tmp_path/"central/checkpoints/latest.pt")
    assert not list(result.glob("*.pt"))
