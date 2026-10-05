from dataclasses import asdict

import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.envs.lqr_env import LQREnv, get_optimal_gain
from csbapr.sindy.feedback import LQRFeedbackReference
from scripts import test_lqr_convergence as runner


@pytest.mark.parametrize("method", runner.LQR_METHODS)
def test_episode_boundary_resume_matches_uninterrupted_training(method, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(runner, "EP_LEN", 6)
    runner.seed_training(7)
    config = runner.make_config(method, "cpu")
    config.hidden_dim = 8
    config.num_critics = 2
    config.batch_size = 4
    config.buffer_size = 64
    config.warmup_steps = 2
    agent = CSBAPRAgent(4, 2, config)
    if method in runner.SYMBOLIC_LQR_METHODS:
        agent.pi_sym_torch = LQRFeedbackReference(get_optimal_gain())
    env = LQREnv(seed=7)
    env.action_space.seed(7)
    protocol = {"method": method, "config": asdict(config)}
    first_return, env_steps = runner.run_episode(env, agent, train=True)
    path = str(tmp_path / "last.pt")
    runner.save_training_checkpoint(agent, env, path, next_ep=1, env_steps=env_steps,
                                    history=[first_return], protocol=protocol)
    expected_return, expected_steps = runner.run_episode(env, agent, train=True, env_steps=env_steps)

    runner.seed_training(999)
    resumed = CSBAPRAgent(4, 2, config)
    resumed_env = LQREnv(seed=999)
    resumed_env.action_space.seed(999)
    next_ep, restored_steps, history = runner.restore_training_checkpoint(
        resumed, resumed_env, path, protocol,
    )
    assert next_ep == 1 and history == [first_return]
    actual_return, actual_steps = runner.run_episode(
        resumed_env, resumed, train=True, env_steps=restored_steps,
    )
    assert actual_return == expected_return and actual_steps == expected_steps
    assert resumed.training_steps == agent.training_steps
    assert resumed.alpha == agent.alpha
    assert resumed.alpha_optimizer.param_groups[0]["params"][0] is resumed.log_alpha
    assert resumed._reward_ema_var == agent._reward_ema_var
    np.testing.assert_array_equal(resumed.belief_tracker.belief, agent.belief_tracker.belief)
    assert vars(resumed.surprise_computer) == vars(agent.surprise_computer)
    for model_name in ("actor", "critic", "target_critic"):
        for actual, expected in zip(getattr(resumed, model_name).parameters(),
                                    getattr(agent, model_name).parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wrong_protocol_is_not_silently_restarted(tmp_path):
    config = runner.make_config("bapr", "cpu")
    agent = CSBAPRAgent(4, 2, config)
    env = LQREnv(seed=0)
    env.action_space.seed(0)
    path = str(tmp_path / "last.pt")
    runner.save_training_checkpoint(agent, env, path, next_ep=0, env_steps=0,
                                    history=[], protocol={"training_seed": 0})
    with pytest.raises(ValueError, match="exact protocol"):
        runner.restore_training_checkpoint(agent, env, path, {"training_seed": 1})


def test_evaluation_snapshot_does_not_duplicate_replay(tmp_path):
    config = runner.make_config("bapr", "cpu")
    agent = CSBAPRAgent(4, 2, config)
    agent.replay_buffer.push(np.zeros(4), np.zeros(2), 0., np.ones(4), False)
    path = tmp_path / "best.pt"
    agent.save(path)
    assert torch.load(path, weights_only=False)["training_state"] is None
