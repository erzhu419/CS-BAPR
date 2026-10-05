from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.envs import nonlinear_load_env as nonlinear
from scripts import test_lqr_convergence as runner
from scripts.preflight_nonlinear_load_v9 import action_for, local_linear_gain, rollout

METHODS = (
    "csbapr-centered-mlp-no-target-shift", "csbapr-centered-mlp-no-sindy-no-target-shift",
    "bapr-no-target-shift", "resac-no-target-shift",
)


@pytest.mark.parametrize("method", METHODS)
def test_nonlinear_presets_keep_frozen_v8_coefficients(method):
    original = asdict(runner.make_config(method, "cpu", "random_load"))
    actual = asdict(runner.make_config(method, "cpu", "nonlinear_load"))
    assert actual == {**original, "env_name": "NonlinearLoad-v0"}
    assert isinstance(runner.make_env("nonlinear_load", 1, 5), nonlinear.NonlinearLoadEnv)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_known_local_feedback_uses_the_frozen_pilot_controller(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    reference = runner.known_reference("nonlinear_load").to(device)
    gain = local_linear_gain()
    observation, _ = nonlinear.NonlinearLoadEnv(50, 50100).reset()
    with torch.no_grad():
        actual = reference(torch.as_tensor(observation, dtype=torch.float32, device=device)).cpu().numpy()
    expected = action_for("local_linear_feedback", observation, gain)
    np.testing.assert_allclose(actual, expected, atol=1e-7)

    def action(observation, deterministic):
        assert deterministic
        with torch.no_grad():
            return reference(torch.as_tensor(observation, dtype=torch.float32, device=device)).cpu().numpy()

    learner_path = runner.evaluate_ood(SimpleNamespace(select_action=action), [1, 50], n_eval=2,
                                       seed_base=50100, environment="nonlinear_load")
    control_path = runner.evaluate_feedback(reference, [1, 50], n_eval=2, seed_base=50100,
                                            environment="nonlinear_load")
    assert learner_path == control_path["amplitude_results"]


def test_inverse_feedback_uses_pilot_physics_and_paired_paths():
    actual = runner.evaluate_inverse_feedback([1, 50], n_eval=2, seed_base=50100)
    for severity in (1, 50):
        expected = [rollout("inverse_feedback", severity, seed, None)["return"]
                    for seed in (50100, 50101)]
        assert actual["amplitude_results"][severity]["returns"] == expected
        assert actual["amplitude_results"][severity]["evaluation_seeds"] == [50100, 50101]
    assert actual["reference_is_optimal"] is False


@pytest.mark.parametrize("method", METHODS)
def test_nonlinear_resume_reproduces_next_episode(method, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(runner, "EP_LEN", 6)
    monkeypatch.setattr(nonlinear, "EP_LEN", 6)
    runner.seed_training(5)
    config = runner.make_config(method, "cpu", "nonlinear_load")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 4
    config.buffer_size, config.warmup_steps = 64, 2
    agent = CSBAPRAgent(6, 2, config)
    if method in runner.SYMBOLIC_LQR_METHODS:
        agent.pi_sym_torch = runner.known_reference("nonlinear_load")
    env = runner.make_env("nonlinear_load", 50, 5)
    env.action_space.seed(5)
    protocol = {"method": method, "config": asdict(config), "evaluation_episodes_per_severity": 100,
                "environment_settings": env.specification()}
    first, steps = runner.run_episode(env, agent, train=True)
    path = str(tmp_path / "last.pt")
    runner.save_training_checkpoint(agent, env, path, next_ep=1, env_steps=steps,
                                    history=[first], protocol=protocol)
    expected, expected_steps = runner.run_episode(env, agent, train=True, env_steps=steps)
    runner.seed_training(999)
    resumed = CSBAPRAgent(6, 2, config)
    resumed_env = runner.make_env("nonlinear_load", 50, 999)
    resumed_env.action_space.seed(999)
    _, restored_steps, _ = runner.restore_training_checkpoint(resumed, resumed_env, path, protocol)
    actual, actual_steps = runner.run_episode(resumed_env, resumed, train=True, env_steps=restored_steps)
    assert actual == expected and actual_steps == expected_steps
    assert resumed.training_steps == agent.training_steps
    for name in ("actor", "critic", "target_critic"):
        for actual, expected in zip(getattr(resumed, name).parameters(), getattr(agent, name).parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(resumed_env.disturbance, env.disturbance)
    with pytest.raises(ValueError, match="exact protocol"):
        runner.restore_training_checkpoint(resumed, resumed_env, path,
                                            {**protocol, "evaluation_episodes_per_severity": 20})
