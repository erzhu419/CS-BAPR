from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v136_warmup_context as worker
from scripts import run_general_ood_v136_multinode as transport
from test_general_ood_v134 import make_agent


def test_predeclared_budget_and_no_early_context_intervention():
    spec = worker.stage_spec(3)
    assert len(worker.cells(spec)) == 100 and 2*2*100 == 400
    assert spec["warmup_steps"] == 200 and spec["first_ood_step"] == 201
    assert spec["intensity"] == 50. and spec["sampled_noise_scale"] == 1.
    assert spec["reward_changes"] == [] and not spec["policy_receives_profile_labels"]
    assert len(worker.cells(worker.stage_spec(4, True))) == 4


@pytest.mark.parametrize("condition", worker.CONDITIONS)
def test_single_predeclared_profile_switch_occurs_at201(condition):
    env = worker.make_env(condition)
    env.unwrapped._terminate_when_unhealthy = False
    env.reset(seed=153000)
    names = []
    for _ in range(202):
        _, _, _, _, info = env.step(np.zeros(3))
        names.append(info["ood_profile"])
        worker.v135.actual_multipliers(env)
    assert names[:200] == ["id"]*200
    assert names[200:] == [condition.removeprefix("abrupt/")]*2
    env.close()


def test_paired_prefix_and_nonzero_context_hold_preserve_weights_and_rng(monkeypatch):
    agent = make_agent()
    before, weights = deepcopy(vars(agent.context_tracker)), deepcopy(agent.actor.state_dict())
    rng = torch.get_rng_state().clone()
    inputs = []
    handle = agent.actor.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("CPU probe must not enumerate CUDA"))
    cell = dict(condition="abrupt/mega_event", mode="sampled", reset_seed=153000, noise_seed=136000)
    pair = worker.evaluate_pair(agent, cell, max_steps=8, warmup_steps=4)
    handle.remove()
    online, held = pair["episodes"]
    assert pair["prefix_equal"] and pair["prefix_steps"] == 5 and pair["prefix_max_abs_difference"] == 0.
    assert online["context_updates"] == online["length"] and held["context_updates"] == 4
    assert held["mean_context_l2"] > 0. and held["mean_post_context_change_l2"] == 0.
    assert online["mean_post_context_change_l2"] > 0.
    assert held["post_warmup_steps"] == held["non_id_steps"] == 4
    context_inputs = torch.cat(inputs[online["length"]:])[:, agent.state_dim:]
    np.testing.assert_array_equal(context_inputs[4:].numpy(), np.broadcast_to(held["context_at_intervention"], (4, agent.context_dim)))
    for key, value in before.items():
        np.testing.assert_equal(value, vars(agent.context_tracker)[key])
    torch.testing.assert_close(weights, agent.actor.state_dict(), rtol=0., atol=0.)
    assert torch.equal(rng, torch.get_rng_state()) and agent.replay_buffer.size == agent.training_steps == 0
    assert pair == worker.evaluate_pair(agent, cell, max_steps=8, warmup_steps=4)


def test_early_failure_or_cutoff_is_retained_without_ood_exposure():
    agent = make_agent()
    cell = dict(condition="abrupt/mega_event", mode="deterministic", reset_seed=153000, noise_seed=None)
    pair = worker.evaluate_pair(agent, cell, max_steps=8)
    for e in pair["episodes"]:
        assert not e["intervention_reached"] and e["context_at_intervention"] is None
        assert e["post_warmup_steps"] == e["post_warmup_return"] == e["non_id_steps"] == 0
        assert e["context_updates"] == e["length"]
    assert pair["prefix_steps"] == 8


def test_changed_prefix_is_rejected(monkeypatch):
    original = worker.evaluate_episode
    def altered(*args, **kw):
        result, prefix = original(*args, **kw)
        if args[2] == "held-context":
            prefix[0, 0] += 1.
        return result, prefix
    monkeypatch.setattr(worker, "evaluate_episode", altered)
    with pytest.raises(AssertionError):
        worker.evaluate_pair(make_agent(), dict(condition="id", mode="deterministic", reset_seed=153000, noise_seed=None), 8, 4)


def test_pair_boundary_resume_is_exact(tmp_path):
    agent = make_agent()
    agent.training_steps = 299745
    spec = dict(worker.stage_spec(3, True), warmup_steps=4, max_steps=8)
    direct = dict(pairs=[])
    worker.collect(agent, spec, direct, lambda: None)
    partial, path = dict(pairs=[]), tmp_path/"progress.json"
    def save():
        path.write_text(json.dumps(partial))
        if len(partial["pairs"]) == 2:
            raise InterruptedError()
    with pytest.raises(InterruptedError):
        worker.collect(agent, spec, partial, save)
    resumed = json.loads(path.read_text())
    worker.collect(agent, spec, resumed, lambda: None)
    assert resumed == direct


def test_transport_copies_final_only_and_spawns_native_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    source = dict(revision="general_ood_v133_source_extension", training_seed=3, arm="mse-control", smoke=False,
                  checkpoint=str(tmp_path/"central/final.pt"), resumable_checkpoint=str(tmp_path/"central/latest.pt"))
    monkeypatch.setattr(transport.subprocess, "run", lambda *args, **kw: SimpleNamespace(stdout=json.dumps(source)))
    copies = []
    def copy(path):
        copies.append(path)
        return tmp_path/"node/final.pt"
    monkeypatch.setattr(transport.transport, "copy_artifact", copy)
    staged = transport.stage_source(3)
    assert copies == [source["checkpoint"]]
    assert json.loads(staged.read_text())["checkpoint"] == str(tmp_path/"node/final.pt")
    monkeypatch.setattr(transport, "stage_source", lambda seed: staged)
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    calls = []
    monkeypatch.setattr(transport.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    result = tmp_path/"node/results"
    result.mkdir(parents=True)
    (result/"warmup_context.json").write_text(json.dumps(dict(source=dict(checkpoint=str(tmp_path/"node/final.pt")))))
    transport.run(SimpleNamespace(seed=3, checkpoint_dir=tmp_path/"node/progress", result_dir=result,
        resume_from=tmp_path/"node/progress/progress.json", smoke=True))
    assert calls[0][1]["env"] == {"NATIVE": "yes"} and "--resume-from" in calls[0][0]
    publications = [command for command, _ in calls if command[0] == "rsync"]
    assert len(publications) == 1 and publications[0][-2].endswith("warmup_context.json")
