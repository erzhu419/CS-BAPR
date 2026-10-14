from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import torch

from scripts import analyze_general_ood_v139_frozen_diagnostics as worker
from scripts import run_general_ood_v139_multinode as transport


def test_age_sampling_and_private_rng():
    before = np.random.get_state()
    lhs, rhs = worker.replay_indices(300000, 128), worker.replay_indices(300000, 128)
    for (lo, hi, a), (_, _, b) in zip(lhs, rhs):
        np.testing.assert_array_equal(a, b)
        assert len(set(a)) == 128 and np.all((a >= lo) & (a < hi))
    np.testing.assert_equal(before, np.random.get_state())


def test_native_actor_density_parity_and_stable_gradient():
    from csbapr.networks.policy import GaussianPolicy
    actor = GaussianPolicy(4, 3, 16)
    state = torch.zeros(12, 4)
    before = torch.get_rng_state()
    action, lp, noise, mean, log_std = actor.evaluate(state)
    torch.set_rng_state(before)
    terms = worker.density_terms(mean, log_std, noise)
    torch.testing.assert_close(terms["action"], action, rtol=0., atol=0.)
    torch.testing.assert_close(terms["native"], lp, rtol=0., atol=0.)
    bounded = worker.density_terms(mean, log_std, noise.clamp(-.5, .5))
    torch.testing.assert_close(bounded["stable"], bounded["native"], rtol=1e-5, atol=1e-5)
    mean = torch.tensor([[20., -20., 0.]], requires_grad=True)
    terms = worker.density_terms(mean, torch.zeros_like(mean), torch.zeros_like(mean))
    native = torch.autograd.grad(terms["native"].sum(), mean, retain_graph=True)[0]
    stable = torch.autograd.grad(terms["stable"].sum(), mean)[0]
    assert native[0, :2].abs().sum() == 0. and stable[0, :2].abs().min() > 1.9


def test_target_decomposition_and_termination_mask():
    reward, done, q, lp = [torch.tensor(x) for x in ([1., 2.], [0., 1.], [100., 200.], [-3., 4.])]
    terms = worker.target_terms(reward, done, q, lp, .2, .99, 2.)
    torch.testing.assert_close(sum(terms.values()), reward/2.+(1.-done)*.99*(q-.2*lp))
    assert terms["entropy"][1] == terms["bootstrap"][1] == 0.


def test_probe_does_not_mutate_agent_rng_or_replay(monkeypatch):
    torch.set_num_threads(1)
    config = worker.v132.v39.v38.make_config("Hopper-v4", "fixed-penalty", "cpu")
    config.state_dim, config.action_dim, config.hidden_dim = 11, 3, 16
    config.weight_reg = 0.
    agent = worker.v132.SourceStabilityAgent(11, 3, config, "mse-control")
    context = np.zeros(agent.context_dim, dtype=np.float32)
    row = (np.zeros(11), np.zeros(3), 1., np.ones(11)*.1, 0., context, context)
    agent.replay_buffer.buffer = [row]*300000
    agent.replay_buffer.position = 300000
    agent._reward_ema_var = 1.
    weights, context_before = deepcopy(agent.actor.state_dict()), deepcopy(vars(agent.context_tracker))
    torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: (_ for _ in ()).throw(AssertionError("no CUDA")))
    monkeypatch.setattr(agent, "update", lambda: (_ for _ in ()).throw(AssertionError("no updates")))
    report = worker.probe_agent(agent, 8)
    assert report["optimizer_updates"] == report["training_interactions"] == 0
    torch.testing.assert_close(weights, agent.actor.state_dict(), rtol=0., atol=0.)
    np.testing.assert_equal(context_before, vars(agent.context_tracker))
    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0., atol=0.)
    np.testing.assert_equal(numpy_rng, np.random.get_state())
    assert python_rng == random.getstate() and agent.replay_buffer.buffer[0] is row
    assert all(p.grad is None for net in (agent.actor, agent.critic, agent.target_critic) for p in net.parameters())


def test_transport_native_cpu_and_no_checkpoint_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    monkeypatch.setattr(transport, "stage_source", lambda case: tmp_path/"source.json")
    calls, published = [], []
    monkeypatch.setattr(transport.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    monkeypatch.setattr(transport, "publish", lambda path: published.append(path))
    args = SimpleNamespace(case=worker.CASES[0], result_dir=tmp_path/"result", smoke=True)
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"} and "--smoke" in calls[0][0]
    assert "--device" not in calls[0][0] and published == [args.result_dir]
