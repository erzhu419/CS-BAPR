from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v143_terminal_trace as worker
from scripts import run_general_ood_v143_multinode as transport


@pytest.mark.parametrize("profile", worker.source_worker.PROFILES)
@pytest.mark.parametrize("noise", (None, 137000))
def test_instrumentation_is_passive_and_preserves_models_context_rng(profile, noise, monkeypatch):
    torch.set_num_threads(1)
    env, _ = worker.source_worker.make_env("id")
    agent = worker.source_worker.make_agent(env, "cpu", "mse-control")
    env.close()
    cell = dict(condition="id" if profile == "id" else "static/"+profile,
                mode="deterministic" if noise is None else "sampled", reset_seed=154000, noise_seed=noise)
    factory = worker.evaluator.make_env if profile == "id" else worker.v138.make_wrapped_env
    before, weights, rng = deepcopy(vars(agent.context_tracker)), deepcopy(agent.actor.state_dict()), torch.get_rng_state()
    canonical = worker.evaluator.evaluate_episode(agent, cell, "online", 256, env_factory=factory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("CPU trace must not enumerate CUDA"))
    observed = worker.evaluate_episode(agent, profile, cell, 256)
    trace = observed.pop("terminal_trace")
    assert observed == canonical and trace["steps"] == observed["length"]
    assert 0 < len(trace["tail"]) <= 17 and trace["tail"][-1]["step"] == observed["length"]
    assert len(trace["tail"][-1]["before"]["qpos"]) == 6
    assert len(trace["tail"][-1]["joint_torque"]) == 3
    assert all(0 <= f <= 1 for f in trace["action_saturation_fraction"])
    np.testing.assert_equal(before, vars(agent.context_tracker))
    torch.testing.assert_close(weights, agent.actor.state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0., atol=0.)
    assert agent.training_steps == agent.replay_buffer.size == 0


def test_cases_include_matched_models_and_cross_physics_positive_controls():
    assert len(worker.CASE_NAMES) == 8 and worker.CASE_NAMES == transport.CASE_NAMES
    assert worker.CASES[-2:] == [(4, "id", "friction-only"), (4, "id", "gravity-only")]


def test_stage_copies_final_only_and_native_publication_is_scalar(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    source = dict(revision=worker.source_worker.REVISION, smoke=False,
        checkpoint=str(tmp_path/"central/policy/final.pt"), resumable_checkpoint=str(tmp_path/"central/policy/latest.pt"))
    calls, copies = [], []
    monkeypatch.setattr(transport.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(stdout=json.dumps(source)))
    monkeypatch.setattr(transport.transport, "copy_artifact", lambda p: copies.append(p) or transport.transport.local_path(p))
    path = transport.stage_source("seed4_id_to_gravity-only")
    assert copies == [source["checkpoint"]] and json.loads(path.read_text())["checkpoint"] == str(tmp_path/"node/policy/final.pt")
    monkeypatch.setattr(transport, "stage_source", lambda case: path)
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    published = []
    monkeypatch.setattr(transport, "publish", lambda p: published.append(p))
    args = SimpleNamespace(case="seed4_id_to_gravity-only", checkpoint_dir=tmp_path/"progress", result_dir=tmp_path/"result",
                           resume_from=tmp_path/"progress/progress.json", smoke=True)
    transport.run(args)
    assert calls[-1][1]["env"] == {"NATIVE": "yes"} and "--resume-from" in calls[-1][0]
    assert published == [args.result_dir] and "--smoke" in calls[-1][0]
