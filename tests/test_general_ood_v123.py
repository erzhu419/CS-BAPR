from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v123_source_extension as worker
from scripts import summarize_general_ood_v123 as merger
from test_general_ood_v121 import make_pair


def test_protocol_extends_every_seed_without_changing_hyperparameters():
    assert worker.stage_spec()["steps"] == 300000 and worker.stage_spec()["source_steps"] == 100000
    assert worker.stage_spec(True)["steps"] == 100512
    assert worker.v121.SEEDS == (0, 1, 2)
    for key, value in worker.v122.stage_spec().items():
        if key not in ("steps", "study_revision", "intervention"):
            assert worker.stage_spec()[key] == value


@pytest.mark.parametrize("cutoff", (11, 12))
def test_prefix_handoff_and_interrupted_extension_match_uninterrupted_training(tmp_path, monkeypatch, cutoff):
    source_spec = dict(worker.v122.stage_spec(), steps=cutoff, checkpoint_interval=4,
        diagnostic_interval=4, evaluation_interval=4, evaluation_episodes=1, evaluation_max_steps=4)
    monkeypatch.setattr(worker.v122, "stage_spec", lambda: source_spec)
    spec = dict(source_spec, steps=24, source_steps=cutoff, study_revision=worker.REVISION,
                intervention="unchanged-v122-training-budget-extension")
    def pair(seed):
        env, agent = make_pair(seed)
        agent.config.weight_reg = 0.
        return env, agent
    env, direct = pair(1)
    expected = worker.v121.train(direct, env, 1, spec, tmp_path/"direct/latest.pt")
    env.close()
    env, prefix = pair(1)
    source = tmp_path/"prefix/latest.pt"
    worker.v121.train(prefix, env, 1, source_spec, source)
    env.close()
    save = worker.v121.save_progress
    def interrupt(agent, env, loop, spec, seed, path):
        save(agent, env, loop, spec, seed, path)
        if loop["step"] == 20:
            raise InterruptedError("after handoff, episode resets, and an additional checkpoint")
    monkeypatch.setattr(worker.v121, "save_progress", interrupt)
    env, extended = pair(99)
    checkpoint = tmp_path/"extension/latest.pt"
    with pytest.raises(InterruptedError):
        worker.continue_training(extended, env, 1, spec, source, checkpoint)
    env.close()
    monkeypatch.setattr(worker.v121, "save_progress", save)
    env, resumed = pair(98)
    actual = worker.continue_training(resumed, env, 1, spec, source, checkpoint, checkpoint)
    assert actual["optimizer_updates"] == expected["optimizer_updates"] == 17
    assert actual["partial_episode_steps"] == expected["partial_episode_steps"]
    assert resumed.replay_buffer.size == direct.replay_buffer.size == 24
    for model in ("actor", "critic", "target_critic"):
        for key, value in getattr(direct, model).state_dict().items():
            assert torch.equal(value, getattr(resumed, model).state_dict()[key])
    for before, after in zip(direct.replay_buffer.buffer, resumed.replay_buffer.buffer):
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
    assert sum(b["interactions"] for b in actual["diagnostic_blocks"]) == 24
    assert torch.load(source, map_location="cpu", weights_only=False)["extra_state"]["loop"]["step"] == cutoff
    env.close()


def test_summary_counts_only_added_training_and_keeps_regressions():
    sources, extensions = [], []
    for seed in (0, 1, 2):
        source = dict(revision=worker.v122.REVISION, smoke=False, training_seed=seed, config={}, runtime={},
            training=dict(interactions=100000), checkpoint_selection="final-not-best",
            final_evaluation=dict(mean_episode_velocity=1., mean_episode_length=1000., mean_return=2000.,
                                  physical_terminations=0, episodes=[dict(seed=150000)]))
        extension = deepcopy(source)
        extension.update(revision=worker.REVISION, actual_new_training_interactions=200000,
            actual_new_optimizer_updates=200000, source_interactions=100000, final_target_scale={},
            training=dict(interactions=300000, optimizer_updates=299745, periodic_evaluations=[]))
        extension["final_evaluation"]["mean_episode_length"] -= seed*100.
        sources.append(source)
        extensions.append(extension)
    result = merger.summarize(sources, extensions)
    assert result["actual_new_training_interactions"] == 600000 and result["reused_source_interactions"] == 300000
    assert result["mean_change"]["mean_episode_length"] == -100.
    assert len(result["per_training_seed"]) == 3
