from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v131_source_diagnostics as probe
from scripts import run_general_ood_v131_multinode as worker
from scripts import summarize_general_ood_v131 as merger


def make_agent():
    from csbapr.agent import CSBAPRAgent
    from csbapr.config import CSBAPRConfig

    config = CSBAPRConfig(state_dim=2, action_dim=1, hidden_dim=8, num_critics=3,
                         device="cpu", use_context=True, use_nau_actor=False,
                         penalty_scale=0., weight_reg=0., weight_sym=0., jac_weight=0.)
    agent = CSBAPRAgent(2, 1, config)
    agent._reward_ema_var = 1.25
    context = np.zeros(agent.context_dim, dtype=np.float32)
    for i in range(8):
        state = np.array([i/10., -.1], dtype=np.float32)
        agent.replay_buffer.buffer.append((state, np.array([.2], dtype=np.float32), float(i%3),
            state+.1, float(i%4 == 0), context.copy(), context.copy()))
    return agent


def test_tail_share_and_empty_replay_group():
    td = torch.zeros(3, 100)
    td[:, 0] = 10.
    summary = probe.residual_summary(td)
    assert summary["mse"] == 1.
    assert summary["top1pct_loss_share"] == 1.
    assert summary["top10pct_loss_share"] == 1.
    assert probe.residual_summary(td[:, :0]) == {"transitions": 0}


def test_nonfinite_diagnostic_is_not_silently_written():
    with pytest.raises(AssertionError):
        probe.describe(torch.tensor([float("nan")]))


def test_native_loss_batch_gradients_and_no_mutation():
    torch.manual_seed(131)
    agent = make_agent()
    before = {name: deepcopy(model.state_dict()) for name, model in
              (("actor", agent.actor), ("critic", agent.critic), ("target", agent.target_critic))}
    ema, training_steps = agent._reward_ema_var, agent.training_steps
    values, metrics = probe.diagnose_batch(agent, np.arange(8))
    assert values["td_error"].shape == (3, 8)
    assert metrics["critic_total"] > 0 and metrics["actor_total"] > 0
    assert torch.all(values["discounted_bootstrap_q"][values["done"].bool()] == 0)
    assert torch.all(values["discounted_entropy"][values["done"].bool()] == 0)
    torch.testing.assert_close(values["target"][:, values["done"].bool()],
        values["normalized_reward"][values["done"].bool()].expand(3, -1))
    assert (agent._reward_ema_var, agent.training_steps) == (ema, training_steps)
    assert all(p.grad is None for p in (*agent.actor.parameters(), *agent.critic.parameters()))
    for name, model in (("actor", agent.actor), ("critic", agent.critic), ("target", agent.target_critic)):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name][key], rtol=0., atol=0.)


def test_pool_noise_reproducibility_batch_concatenation_and_rng_restore(monkeypatch):
    torch.manual_seed(132)
    agent = make_agent()
    def unexpected_cuda_probe():
        raise AssertionError("CPU diagnosis must not query CUDA device count")
    monkeypatch.setattr(torch.cuda, "device_count", unexpected_cuda_probe)
    rng = torch.get_rng_state().clone()
    row = probe.diagnose_pool(agent, np.arange(8), 13100, batch_size=4)
    assert torch.equal(rng, torch.get_rng_state())
    assert row == probe.diagnose_pool(agent, np.arange(8), 13100, batch_size=4)
    assert row["samples"] == 8 and row["batches"] == 2
    assert row["replay_done"]["transitions"] == 2
    assert row["replay_not_done"]["transitions"] == 6
    assert len(row["per_critic"]) == 3
    assert row["scales"]["predicted_q"]["count"] == 24
    assert len(row["worst_transitions"]) == 5
    assert row["native_critic_loss_reconstructed"]


@pytest.mark.parametrize("seed,steps,relative", [
    (1, 300000, "experiments/general_ood_v123_source_extension_20261011/runtime"),
    (3, 100000, "experiments/general_ood_v130_independent_replication_20261011/runtime/prefix"),
    (4, 300000, "experiments/general_ood_v130_independent_replication_20261011/runtime")])
def test_source_paths_use_actual_prefix_or_source_lineage(seed, steps, relative):
    assert worker.source_path(seed, steps) == worker.transport.CENTRAL_PROJECT/relative/f"run_results/seed{seed}/source.json"


def test_new_node_stages_full_state_and_uses_native_libraries(tmp_path, monkeypatch):
    calls, stages = [], []
    monkeypatch.setattr(worker.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(worker.transport, "LOCAL_PROJECT", tmp_path/"node")
    monkeypatch.setattr(worker.transport, "CENTRAL_RUNTIME", tmp_path/"central"/worker.transport.RUNTIME_REL)
    monkeypatch.setattr(worker.transport, "stage_source", lambda path, full: stages.append((path, full)))
    monkeypatch.setattr(worker.transport, "native_environment", lambda: {"NATIVE": "yes"})
    result = tmp_path/"node/result"
    result.mkdir(parents=True)
    (result/"source_diagnostics.json").write_text('{"checkpoint":"'+str(tmp_path/"node/latest.pt")+'"}')
    monkeypatch.setattr(worker.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    worker.run(SimpleNamespace(seed=3, source_steps=100000, result_dir=result, smoke=True))
    assert stages == [(worker.source_path(3, 100000), True)]
    command, options = calls[0]
    assert "--smoke" in command and options["env"] == {"NATIVE": "yes"}
    assert "--source-json" in command
    assert not any(part == "rsync" for command, _ in calls for part in command)


def write_formal_results(root):
    for seed, age in probe.SOURCE_CELLS:
        cells = []
        for noise in probe.NOISE_SEEDS:
            cells.append(dict(noise_seed=noise, samples=4096, batches=16,
                residuals=dict(transitions=4096, mse=1., abs_td={"p50": .5, "p99": 2.},
                               top1pct_loss_share=.2, top10pct_loss_share=.5),
                replay_done=dict(transitions=96, mse=2.),
                replay_not_done=dict(transitions=4000), per_critic=[{}]*5,
                native_critic_loss_reconstructed=True,
                scales={key: {"mean": 1., "p99": 2.} for key in
                    ("predicted_q", "target", "discounted_bootstrap_q", "discounted_entropy",
                     "normalized_reward", "policy_log_prob", "policy_std_mean")},
                batch_metrics={"critic_total": {"mean": 1., "max": 2.}}))
        row = dict(revision=probe.REVISION, training_seed=seed, source_steps=age, smoke=False,
            optimizer_updates=0, environment_step_calls=0, new_training_interactions=0,
            gradient_evaluation_only=True, source_readout_refitted=False,
            checkpoint_selection="predeclared-final-full-state", noise_seeds=list(probe.NOISE_SEEDS),
            runtime={"torch": "native", "device": "cpu", "cuda_initialized": False},
            alpha=.2, checkpoint="server/latest.pt",
            pools={name: deepcopy(cells) for name in ("uniform_training", "last10k")})
        directory = root/f"seed{seed}_age{age}"
        directory.mkdir()
        (directory/"source_diagnostics.json").write_text(json.dumps(row))


def test_merger_requires_all_six_predeclared_sources(tmp_path):
    write_formal_results(tmp_path)
    result = merger.summarize(tmp_path)
    assert result["source_states"] == 6 and result["diagnostic_cells"] == 36
    assert [(r["training_seed"], r["source_steps"]) for r in result["records"]] == list(probe.SOURCE_CELLS)
    assert result["records"][0]["pools"]["last10k"]["replay_done_loss_share"] == 192/4096
    (tmp_path/"seed4_age300000/source_diagnostics.json").unlink()
    with pytest.raises(FileNotFoundError):
        merger.summarize(tmp_path)


@pytest.mark.parametrize("key,value", [("smoke", True), ("optimizer_updates", 1),
                                       ("noise_seeds", [13100])])
def test_merger_rejects_smoke_training_or_incomplete_noise_grid(tmp_path, key, value):
    write_formal_results(tmp_path)
    path = tmp_path/"seed3_age100000/source_diagnostics.json"
    row = json.loads(path.read_text())
    row[key] = value
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError):
        merger.summarize(tmp_path)
