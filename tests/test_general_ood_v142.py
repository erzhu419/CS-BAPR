from copy import deepcopy
import json
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v142_budget_extension as worker
from scripts import run_general_ood_v142_multinode as transport


def test_budget_only_keeps_rates_physics_and_evaluation():
    for profile in worker.PROFILES:
        source, spec = worker.source_worker.stage_spec(profile), worker.stage_spec(profile)
        assert spec["steps"] == 600000 and spec["source_steps"] == 300000
        assert worker.stage_spec(profile, True)["steps"] == 300512
        for key, value in source.items():
            if key not in ("steps", "study_revision"):
                assert spec[key] == value
    assert sum(worker.stage_spec(p)["steps"]-300000 for p in worker.PROFILES)*2 == 1800000


@pytest.mark.parametrize("profile", worker.PROFILES)
@pytest.mark.parametrize("cutoff", (11, 12))
def test_full_handoff_mid_episode_or_timeout_then_resume(profile, cutoff, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    def pair(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env, _ = worker.make_env(profile)
        env._max_episode_steps = 3
        agent = worker.make_agent(env, "cpu", "mse-control")
        agent.config.batch_size = 8
        return env, agent
    source_spec = dict(worker.source_worker.stage_spec(profile, True), steps=cutoff,
        checkpoint_interval=4, diagnostic_interval=4, evaluation_interval=4, evaluation_max_steps=4)
    spec = dict(source_spec, steps=24, source_steps=cutoff, study_revision=worker.REVISION,
                intervention="training-budget-only")
    callback = lambda a, seed, episodes, limit: worker.v137.evaluate(a, profile, seed, episodes, limit, env_factory=worker.make_env)
    env, direct = pair(3)
    expected = worker.v132.train(direct, env, 3, spec, tmp_path/"direct/latest.pt", evaluation_fn=callback)
    expected_rng = torch.get_rng_state()
    env.close()
    env, prefix = pair(3)
    source = tmp_path/"prefix/latest.pt"
    worker.v132.train(prefix, env, 3, source_spec, source, evaluation_fn=callback)
    env.close()
    assert (cutoff == 12) == (torch.load(source, weights_only=False)["extra_state"]["loop"]["episode_age"] == 0)
    save = worker.v132.save_progress
    def interrupt(*args):
        save(*args)
        if args[2]["step"] == 20:
            raise InterruptedError()
    monkeypatch.setattr(worker.v132, "save_progress", interrupt)
    env, partial = pair(99)
    checkpoint = tmp_path/"extension/latest.pt"
    with pytest.raises(InterruptedError):
        worker.continue_training(partial, env, 3, spec, source_spec, source, checkpoint, evaluation_fn=callback)
    env.close()
    monkeypatch.setattr(worker.v132, "save_progress", save)
    env, resumed = pair(98)
    actual = worker.continue_training(resumed, env, 3, spec, source_spec, source, checkpoint, checkpoint, callback)
    assert actual["optimizer_updates"] == expected["optimizer_updates"] == 17
    assert actual["periodic_evaluations"] == expected["periodic_evaluations"]
    assert [x["step"] for x in actual["periodic_evaluations"]].count(12) == 1
    assert actual["partial_episode_steps"] == expected["partial_episode_steps"]
    for name in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        torch.testing.assert_close(getattr(direct, name).state_dict(), getattr(resumed, name).state_dict(), rtol=0., atol=0.)
    torch.testing.assert_close(expected_rng, torch.get_rng_state(), rtol=0., atol=0.)
    assert direct.alpha == resumed.alpha and direct._reward_ema_var == resumed._reward_ema_var
    np.testing.assert_equal(vars(direct.context_tracker), vars(resumed.context_tracker))
    np.testing.assert_equal(vars(direct.belief_tracker), vars(resumed.belief_tracker))
    for lhs, rhs in zip(direct.replay_buffer.buffer, resumed.replay_buffer.buffer):
        for a, b in zip(lhs, rhs):
            np.testing.assert_array_equal(a, b)
    saved = torch.load(source, weights_only=False)["extra_state"]
    assert saved["spec"] == source_spec and saved["loop"]["step"] == cutoff
    assert all(g["lr"] == 3e-5 for g in resumed.actor_optimizer.param_groups)
    assert all(g["lr"] == 3e-4 for g in resumed.critic_optimizer.param_groups)
    env.close()


def test_stage_reads_only_latest_and_rejects_wrong_source(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    source = dict(revision=worker.source_worker.REVISION, training_seed=3, training_profile="id", smoke=False,
                  resumable_checkpoint=str(tmp_path/"central/prefix/latest.pt"))
    monkeypatch.setattr(transport.subprocess, "run", lambda cmd, **kw: SimpleNamespace(stdout=json.dumps(source)))
    copies = []
    monkeypatch.setattr(transport.transport, "copy_artifact", lambda p: copies.append(p) or transport.transport.local_path(p))
    path = transport.stage_source(3, "id")
    assert copies == [source["resumable_checkpoint"]]
    assert json.loads(path.read_text())["resumable_checkpoint"] == str(tmp_path/"node/prefix/latest.pt")
    with pytest.raises(AssertionError):
        transport.stage_source(4, "id")
    assert len(copies) == 1


def test_native_spawn_and_source_lineage_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.transport, "CENTRAL_PROJECT", tmp_path/"central")
    monkeypatch.setattr(transport.transport, "LOCAL_PROJECT", tmp_path/"node")
    monkeypatch.setattr(transport.transport, "native_environment", lambda: {"NATIVE": "yes"})
    monkeypatch.setattr(transport, "stage_source", lambda seed, profile: tmp_path/"node/input/source.json")
    calls, published = [], []
    monkeypatch.setattr(transport.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    result = tmp_path/"node/result"
    result.mkdir(parents=True)
    (result/"source.json").write_text(json.dumps(dict(source_checkpoint=str(tmp_path/"node/prefix/latest.pt"))))
    monkeypatch.setattr(transport.v132, "publish", lambda a: published.append(deepcopy(json.loads((a.result_dir/"source.json").read_text()))))
    args = SimpleNamespace(seed=3, profile="id", checkpoint_dir=tmp_path/"node/ckpt", result_dir=result,
                           resume_from=tmp_path/"node/ckpt/latest.pt", smoke=True)
    transport.run(args)
    assert calls[0][1]["env"] == {"NATIVE": "yes"} and "--resume-from" in calls[0][0] and "--smoke" in calls[0][0]
    assert published[0]["source_checkpoint"] == str(tmp_path/"central/prefix/latest.pt")
    assert published[0]["source_json"] == str(transport.source_path(3, "id"))
    assert not list(result.glob("*.pt"))
