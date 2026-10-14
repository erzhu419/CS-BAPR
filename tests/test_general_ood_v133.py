from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v133_source_extension as worker
from scripts import run_general_ood_v133_multinode as transport
from test_general_ood_v132 import make_pair


def test_budget_only_protocol():
    for arm in worker.v132.ARMS:
        source, spec = worker.v132.stage_spec(arm), worker.stage_spec(arm)
        assert spec["steps"] == 300000 and spec["source_steps"] == 100000
        assert worker.stage_spec(arm, True)["steps"] == 100512
        for key, value in source.items():
            if key not in ("steps", "study_revision"):
                assert spec[key] == value
    assert sum(worker.stage_spec(arm)["steps"]-100000 for arm in worker.v132.ARMS)*2 == 800000


@pytest.mark.parametrize("arm", worker.v132.ARMS)
@pytest.mark.parametrize("cutoff", (11, 12))
def test_handoff_and_resume_match_direct_mid_episode_and_timeout(tmp_path, monkeypatch, arm, cutoff):
    original_spec = worker.v132.stage_spec
    def source_spec(current_arm, smoke=False):
        return dict(original_spec(current_arm, True), steps=cutoff, checkpoint_interval=4,
                    diagnostic_interval=4, evaluation_interval=4, evaluation_max_steps=4)
    monkeypatch.setattr(worker.v132, "stage_spec", source_spec)
    spec = dict(source_spec(arm), steps=24, source_steps=cutoff, study_revision=worker.REVISION,
                intervention="training-budget-only")
    env, direct = make_pair(3, arm)
    expected = worker.v132.train(direct, env, 3, spec, tmp_path/"direct/latest.pt")
    env.close()
    env, prefix = make_pair(3, arm)
    source = tmp_path/"prefix/latest.pt"
    worker.v132.train(prefix, env, 3, source_spec(arm), source)
    env.close()
    assert (cutoff == 12) == (torch.load(source, weights_only=False)["extra_state"]["loop"]["episode_age"] == 0)
    save = worker.v132.save_progress
    def interrupt(agent, env, loop, spec, seed, path):
        save(agent, env, loop, spec, seed, path)
        if loop["step"] == 20:
            raise InterruptedError()
    monkeypatch.setattr(worker.v132, "save_progress", interrupt)
    env, partial = make_pair(99, arm)
    checkpoint = tmp_path/"extension/latest.pt"
    with pytest.raises(InterruptedError):
        worker.continue_training(partial, env, 3, spec, source, checkpoint)
    env.close()
    monkeypatch.setattr(worker.v132, "save_progress", save)
    env, resumed = make_pair(98, arm)
    actual = worker.continue_training(resumed, env, 3, spec, source, checkpoint, checkpoint)
    assert actual["optimizer_updates"] == expected["optimizer_updates"] == 17
    assert actual["partial_episode_steps"] == expected["partial_episode_steps"]
    assert sum(b["interactions"] for b in actual["diagnostic_blocks"]) == 24
    for name in ("actor", "critic", "target_critic"):
        torch.testing.assert_close(getattr(direct, name).state_dict(), getattr(resumed, name).state_dict(), rtol=0., atol=0.)
    for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        torch.testing.assert_close(getattr(direct, name).state_dict(), getattr(resumed, name).state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(direct.log_alpha, resumed.log_alpha, rtol=0., atol=0.)
    assert direct.alpha == resumed.alpha
    assert direct._reward_ema_var == resumed._reward_ema_var
    for key, value in vars(direct.context_tracker).items():
        np.testing.assert_equal(value, vars(resumed.context_tracker)[key])
    for left, right in zip(direct.replay_buffer.buffer, resumed.replay_buffer.buffer):
        for lhs, rhs in zip(left, right):
            np.testing.assert_array_equal(lhs, rhs)
    assert direct.replay_buffer.size == resumed.replay_buffer.size == 24
    assert torch.load(source, weights_only=False)["extra_state"]["loop"]["step"] == cutoff
    with pytest.raises(AssertionError):
        worker.continue_training(resumed, env, 3, dict(spec, arm="huber10" if arm == "mse-control" else "mse-control"), source, checkpoint)
    env.close()


def test_source_stage_copies_full_state_only_and_checks_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    source = dict(revision=worker.v132.REVISION, training_seed=4, arm="huber10", smoke=False,
        resumable_checkpoint=str(tmp_path/"central/source/latest.pt"), checkpoint=str(tmp_path/"central/source/final.pt"))
    calls, copies = [], []
    def response(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(source))
    monkeypatch.setattr(transport.subprocess, "run", response)
    monkeypatch.setattr(transport.transport, "copy_artifact", lambda p: copies.append(p))
    path = transport.stage_source(4, "huber10")
    assert copies == [str(tmp_path/"central/source/latest.pt")]
    assert json.loads(path.read_text())["resumable_checkpoint"] == str(tmp_path/"node/source/latest.pt")
    assert "seed4_huber10/source.json" in calls[0][-1]
    with pytest.raises(AssertionError):
        transport.stage_source(4, "mse-control")
    assert len(copies) == 1


def test_native_spawn_resume_and_authoritative_source_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    source = tmp_path/"node/input/source.json"
    monkeypatch.setattr(transport, "stage_source", lambda seed, arm: source)
    calls, published = [], []
    monkeypatch.setattr(transport.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    result = tmp_path/"node/results"
    result.mkdir(parents=True)
    (result/"source.json").write_text(json.dumps(dict(source_json=str(source), source_checkpoint=str(tmp_path/"node/source/latest.pt"))))
    monkeypatch.setattr(transport.v132, "publish", lambda args: published.append(deepcopy(json.loads((args.result_dir/"source.json").read_text()))))
    args = SimpleNamespace(seed=4, arm="huber10", checkpoint_dir=tmp_path/"node/checkpoints", result_dir=result,
                           resume_from=tmp_path/"node/checkpoints/latest.pt", smoke=True)
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"}
    assert "--resume-from" in calls[0][0] and "--smoke" in calls[0][0]
    assert published[0]["source_json"] == str(transport.source_path(4, "huber10"))
    assert published[0]["source_checkpoint"] == str(tmp_path/"central/source/latest.pt")
    assert not list(result.glob("*.pt"))
