from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from csbapr.losses.jacobian import action_jacobian
from csbapr.sindy.feedback import LQRFeedbackReference
from scripts import test_lqr_convergence as training
from scripts.diagnose_nonlinear_load_v9 import control_action, control_derivatives
from scripts.diagnose_nonlinear_load_v14 import rollout
from scripts.summarize_nonlinear_load_v35 import summarize
from scripts.train_nonlinear_load_v35 import ReferenceInterventionAgent, make_config, make_env, train_episode
from test_nonlinear_load_symbolic_v31 import make_agent as original_agent
from test_nonlinear_load_critic_replay_v25 import parameters_equal


def make_agent(reference):
    original = original_agent("on")
    training.seed_training(19)
    agent = ReferenceInterventionAgent(deepcopy(original.config), 19, reference)
    gain = original.pi_sym_torch.gain.tolist()
    agent.reference_identification = {"gains": {"cubic-local": gain,
        "cost-linear": (original.pi_sym_torch.gain * .6).tolist()}}
    agent.set_reference(reference)
    agent.replay_buffer.buffer = deepcopy(original.replay_buffer.buffer)
    agent.replay_buffer.position = original.replay_buffer.position
    return agent


def test_clip_action_and_all_six_jacobian_columns_match_the_frozen_map():
    gain = np.array([[1., .2, .3, .4, .5, .6], [.7, 1., .6, .5, .4, .3]])
    reference = LQRFeedbackReference(gain, "clip").double()
    states = torch.tensor([[.02] * 6, [4., -1., .1, .2, .3, .4]], dtype=torch.double, requires_grad=True)
    actions = reference(states)
    jacobian = action_jacobian(actions, states, False).detach().numpy()
    actual_gain = reference.gain.detach().numpy()
    linear = -states.detach().numpy() @ actual_gain.T
    np.testing.assert_allclose(actions.detach().numpy(), np.clip(linear, -1, 1), rtol=0, atol=0)
    expected = np.where((np.abs(linear) < 1)[..., None], -actual_gain[None], 0.)
    np.testing.assert_allclose(jacobian, expected, rtol=0, atol=0)
    for column in range(6):
        plus, minus = states.detach().clone(), states.detach().clone()
        plus[:, column] += 1e-6
        minus[:, column] -= 1e-6
        derivative = (reference(plus) - reference(minus)) / 2e-6
        np.testing.assert_allclose(derivative.detach().numpy(), expected[:, :, column], atol=1e-9)
    assert not np.isfinite(reference.compute_L_eff())
    assert np.isfinite(LQRFeedbackReference(gain).compute_L_eff())


def test_evaluation_and_alignment_use_clip_without_changing_legacy_tanh():
    agent = make_agent("cost-linear")
    gain, reference = agent.pi_sym_torch.gain, agent.pi_sym_torch
    states = np.full((2, 6), .5, dtype=np.float32)
    expected = reference(torch.as_tensor(states)).detach().numpy()
    legacy = control_action(agent.actor, {"fitted": gain}, "fitted", states)
    actual = control_action(agent.actor, {"fitted": gain}, "fitted", states, references={"fitted": reference})
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(legacy, expected)
    _, jacobian, _ = control_derivatives(agent.actor, {"fitted": gain}, "fitted", states,
                                         references={"fitted": reference})
    assert jacobian.shape == (2, 2, 6)
    clip, _ = rollout(agent.actor, {"fitted": gain}, "fitted", .5, 50, [66500],
                      env_factory=make_env, references={"fitted": reference})
    tanh, _ = rollout(agent.actor, {"fitted": gain}, "fitted", .5, 50, [66500], env_factory=make_env)
    assert clip["returns"] != tanh["returns"]


@pytest.mark.parametrize("reference", ("cubic-local", "cost-linear"))
def test_checkpoint_preserves_mapping_and_exact_next_update(reference, tmp_path):
    torch.set_num_threads(1)
    agent = make_agent(reference)
    training.seed_training(23)
    agent.update()
    path = str(tmp_path / "last.pt")
    agent.save(path)
    expected = agent.update()
    resumed = make_agent(reference)
    resumed.load(path)
    assert resumed.pi_sym_torch.action_mapping == agent.pi_sym_torch.action_mapping
    assert resumed.update() == expected
    parameters_equal(agent, resumed)
    assert agent.sampling_state() == resumed.sampling_state()
    assert agent.critic_sampler_state() == resumed.critic_sampler_state()
    assert agent.actor_sampler_state() == resumed.actor_sampler_state()
    assert agent.alpha == resumed.alpha
    with pytest.raises(ValueError, match="reference differs"):
        make_agent("cost-linear" if reference == "cubic-local" else "cubic-local").load(path)


def test_existing_tanh_checkpoints_restore_without_a_mapping_field(tmp_path):
    agent = make_agent("cubic-local")
    path = str(tmp_path / "legacy.pt")
    agent.save(path)
    data = torch.load(path, weights_only=False)
    del data["symbolic_reference_mapping"]
    torch.save(data, path)
    restored = make_agent("cubic-local")
    restored.load(path)
    assert restored.pi_sym_torch.action_mapping == "tanh"


def test_prefix_forks_full_state_then_changes_only_reference(tmp_path):
    torch.set_num_threads(1)
    training.seed_training(21)
    config = make_config("off", "cpu")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 8
    config.buffer_size = 10000
    prefix = ReferenceInterventionAgent(config, 21, constraint="off", critic_mode="uniform")
    gain = training.known_reference("nonlinear_load").gain
    prefix.reference_identification = {"gains": {"cubic-local": gain.tolist(), "cost-linear": (gain * .6).tolist()}}
    prefix.set_reference("cubic-local")
    env = make_env(1, 21)
    env.action_space.seed(21)
    history, steps = [], 0
    for episode in range(5):
        reward, steps = train_episode(env, prefix, "balanced", episode, steps)
        history.append(reward)
        prefix.record_temperature(episode)
    path = str(tmp_path / "prefix.pt")
    protocol = {"phase": "prefix", "config": asdict(config)}
    training.save_training_checkpoint(prefix, env, path, next_ep=5, env_steps=steps, history=history, protocol=protocol)
    agents, metrics = [], []
    for reference in ("cubic-local", "cost-linear"):
        arm = ReferenceInterventionAgent(deepcopy(config), 21, constraint="off", critic_mode="uniform")
        assert training.restore_training_checkpoint(arm, env, path, protocol) == (5, 1000, history)
        parameters_equal(prefix, arm)
        arm.critic_sampling = "early"
        arm.set_constraint("on")
        arm.set_reference(reference)
        metrics.append(arm.update())
        agents.append(arm)
    old, new = agents
    assert old.config == new.config and old.alpha == new.alpha
    assert metrics[0]["q_loss"] == metrics[1]["q_loss"]
    for name in ("critic", "target_critic"):
        for a, b in zip(getattr(old, name).parameters(), getattr(new, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert old.sampling_state() == new.sampling_state()
    assert old.critic_sampler_state() == new.critic_sampler_state()
    assert old.actor_sampler_state() == new.actor_sampler_state()
    assert any(not torch.equal(a, b) for a, b in zip(old.actor.parameters(), new.actor.parameters()))


@pytest.mark.parametrize("failure", (None, "total", "early", "small", "fitted"))
def test_unchanged_gates_keep_individual_seed_failures(failure):
    runs = {(reference, seed): {"radius_results": {
        str(radius): {str(load): {"controllers": {"actor": {"mean_cost": 1. if reference == "cost-linear" else 2.,
                                                          "mean_early_state_cost": 1. if reference == "cost-linear" else 2.}},
                                 "paired_excess_costs": {"fitted": {"mean_total_cost": -.1}}}
                      for load in (1, 50)} for radius in (.2, .5, 1.)}}
            for reference in ("cubic-local", "cost-linear") for seed in (21, 22)}
    block = runs["cost-linear", 22]["radius_results"]
    if failure in ("total", "early"):
        block["1.0"]["50"]["controllers"]["actor"]["mean_cost" if failure == "total" else "mean_early_state_cost"] = 2.
    elif failure == "small":
        block["0.2"]["50"]["controllers"]["actor"]["mean_cost"] = 2.01
    elif failure == "fitted":
        block["0.5"]["1"]["paired_excess_costs"]["fitted"]["mean_total_cost"] = 0.
    assert summarize(runs, (21, 22))["decision"]["ready_for_learner_replication"] == (failure is None)
