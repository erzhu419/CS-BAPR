import math

import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
from csbapr.envs.lqr_env import LQREnv, A, B, Q, R, ACTION_SCALE, get_optimal_gain
from csbapr.losses.jacobian import compute_jacobian_loss, compute_jacobian_loss_nD
from csbapr.losses.ood_bound import compute_ood_bound
from csbapr.networks.nau_nmu import NAU_NMU_Actor
from csbapr.sindy.data_collector import collect_trajectories
from csbapr.sindy.feedback import LQRFeedbackReference
from csbapr.sindy.torch_wrapper import SINDyTorchWrapper
from csbapr.sindy.world_model import SymbolicWorldModel
from csbapr.irm.causal_filter import select_best_sindy_coefficients


def linear_fit():
    rng = np.random.default_rng(17)
    states = rng.normal(size=(500, 4))
    controls = rng.uniform(-1, 1, (500, 2))
    targets = states @ (A - np.eye(4)).T + controls @ (B * ACTION_SCALE).T
    return SymbolicWorldModel(4, 2, 1, 0.001).fit(states, controls, targets)


def test_controlled_fit_identifies_separate_A_B():
    model = linear_fit()
    a_fit, b_fit = model.linear_dynamics()
    np.testing.assert_allclose(a_fit, A, atol=1e-10)
    np.testing.assert_allclose(b_fit, B * ACTION_SCALE, atol=1e-10)
    with pytest.raises(ValueError, match="requires actions"):
        model.predict(np.zeros((2, 4)))


def test_episode_boundaries_and_actions_are_aligned():
    env = LQREnv()
    x, u, _ = collect_trajectories(env, n_episodes=3, max_steps=30, seed=0)
    model = SymbolicWorldModel(4, 2, 1, 0.001).fit(x, u, multiple_trajectories=True)
    a_fit, b_fit = model.linear_dynamics()
    np.testing.assert_allclose(a_fit, A, atol=2e-6)
    np.testing.assert_allclose(b_fit, B * ACTION_SCALE, atol=2e-6)
    x2, u2, _ = collect_trajectories(env, n_episodes=3, max_steps=30, seed=0)
    np.testing.assert_array_equal(x[0], x2[0])
    np.testing.assert_array_equal(u[0], u2[0])
    with pytest.raises(ValueError, match="aligned action"):
        model.fit(x, [a[:-1] for a in u], multiple_trajectories=True)


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_wrapper_matches_library_and_control_gradients_at_zero(degree):
    rng = np.random.default_rng(3)
    x, u = rng.normal(size=(200, 2)), rng.normal(size=(200, 1))
    y = np.column_stack([x[:, 0] + 0.5 * u[:, 0], x[:, 1] - u[:, 0]])
    model = SymbolicWorldModel(2, 1, degree, 0.01).fit(x, u, y)
    wrapper = SINDyTorchWrapper(model).double()
    x_test = torch.tensor([[0., 0.], [0.3, -0.2]], dtype=torch.double, requires_grad=True)
    u_test = torch.tensor([[0.], [0.5]], dtype=torch.double, requires_grad=True)
    np.testing.assert_allclose(wrapper(x_test, u_test).detach(),
                               model.predict(x_test.detach().numpy(), u_test.detach().numpy()), atol=1e-6)
    control_grad = torch.autograd.grad(wrapper(x_test, u_test).sum(), u_test)[0]
    torch.testing.assert_close(control_grad, torch.full_like(control_grad, -0.5))
    assert not list(wrapper.parameters())


def test_full_jacobian_does_not_cancel_output_rows():
    policy = torch.nn.Linear(1, 2, bias=False)
    reference = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        policy.weight.copy_(torch.tensor([[1.], [-1.]]))
        reference.weight.zero_()
    reference.requires_grad_(False)
    states = torch.zeros(3, 1)
    loss = compute_jacobian_loss(policy, reference, states)
    torch.testing.assert_close(loss, torch.tensor(2.))
    torch.testing.assert_close(loss, compute_jacobian_loss_nD(policy, reference, states))
    loss.backward()
    assert policy.weight.grad.abs().sum() > 0
    assert reference.weight.grad is None
    with pytest.raises(ValueError, match="action-space shape"):
        compute_jacobian_loss(policy, torch.nn.Linear(1, 3), states)


def test_normalized_squashed_policy_jacobian():
    class Actor(torch.nn.Module):
        action_range = 2.0

        def forward(self, states):
            return states, torch.zeros_like(states)

    reference = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        reference.weight.fill_(2)
    assert compute_jacobian_loss(Actor(), reference, torch.zeros(2, 1)) == 0
    assert compute_jacobian_loss(Actor(), reference, torch.ones(2, 1)) > 1


def test_smooth_whole_actor_bound_includes_all_branches():
    torch.manual_seed(4)
    actor = NAU_NMU_Actor(3, 2, 4, activation="tanh").double()
    bounds = actor.derivative_bounds()
    assert bounds["L_G"] > 0 and bounds["L_H"] > 0
    assert math.isfinite(actor.compute_L_eff())
    action_fn = lambda x: torch.tanh(actor(x.unsqueeze(0))[0]).squeeze(0)
    points = torch.randn(12, 3, dtype=torch.double) * 10
    jacobians = torch.vmap(torch.func.jacrev(action_fn))(points)
    for index in range(11):
        drift = torch.linalg.matrix_norm(jacobians[index + 1] - jacobians[index], ord=2)
        assert drift <= bounds["L_pi"] * (points[index + 1] - points[index]).norm()
    with torch.no_grad():
        actor.nmu_head.coeff.zero_()
    assert actor.compute_L_eff() > 0  # tanh features and squashing still have curvature
    actor.mean_clip = 5
    assert math.isinf(actor.compute_L_eff())
    assert math.isinf(NAU_NMU_Actor(3, 2, 4).compute_L_eff())


def test_reference_uses_fitted_input_and_normalized_cost():
    reference = LQRFeedbackReference.from_world_model(linear_fit(), Q, R * ACTION_SCALE ** 2)
    np.testing.assert_allclose(reference.gain.numpy(), get_optimal_gain(), atol=1e-5)
    assert reference(torch.zeros(7, 4)).shape == (7, 2)
    assert not list(reference.parameters())
    assert math.isfinite(reference.compute_L_eff())


def test_multi_env_selector_keeps_a_fitted_candidate():
    data = {}
    for seed in (7, 9):
        data[str(seed)] = collect_trajectories(LQREnv(), n_episodes=3, max_steps=30, seed=seed)
    template = SymbolicWorldModel(4, 2, 1, 0.001)
    coeffs, variance = select_best_sindy_coefficients(template, data, verbose=False)
    candidates = [SymbolicWorldModel(4, 2, 1, 0.001).fit(x, u, multiple_trajectories=True)
                  for x, u, _ in data.values()]
    assert any(np.array_equal(coeffs, candidate.coeffs) for candidate in candidates)
    assert math.isfinite(variance)


def test_boundary_certificate_not_replay_mean():
    actor = NAU_NMU_Actor(4, 2, 4, activation="tanh")
    reference = LQRFeedbackReference.from_world_model(linear_fit(), Q, R * ACTION_SCALE ** 2)
    result = compute_ood_bound(actor, reference, torch.zeros(4), torch.ones(4))
    assert result["certified"] and result["actual_output_error"] <= result["bound"]
    expected = result["delta"] + result["epsilon"] * result["d"]
    expected += 0.5 * (result["L_eff"] + result["M"]) * result["d"] ** 2
    assert result["bound"] == pytest.approx(expected)
    bad = compute_ood_bound(NAU_NMU_Actor(4, 2, 4), reference, torch.zeros(4), torch.ones(4))
    assert not bad["certified"] and bad["bound"] is None


def test_controlled_reference_training_and_resume(tmp_path):
    torch.set_num_threads(1)
    cfg = CSBAPRConfig(hidden_dim=8, num_critics=3, batch_size=8,
                      sindy_lib_degree=1, sindy_threshold=0.001,
                      sindy_n_explore_episodes=3, max_steps_per_episode=30,
                      actor_type="smooth-nau", device="cpu")
    agent = CSBAPRAgent(4, 2, cfg)
    env = LQREnv(seed=7)
    env.action_space.seed(7)
    builder = lambda model: LQRFeedbackReference.from_world_model(model, Q, R * ACTION_SCALE ** 2)
    agent.sindy_preidentify(env, reference_builder=builder)
    state, _ = env.reset(seed=7)
    for _ in range(16):
        action = env.action_space.sample()
        next_state, reward, term, trunc, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, term or trunc)
        state = next_state
    for _ in range(3):
        metrics = agent.update()
        assert all(math.isfinite(metrics[key]) for key in ["q_loss", "policy_loss", "jac_loss", "L_eff"])
    path = tmp_path / "small_test.pt"
    agent.save(path)
    loaded = CSBAPRAgent(4, 2, cfg)
    loaded.load(path)
    torch.testing.assert_close(loaded.pi_sym_torch.gain, agent.pi_sym_torch.gain)
    np.testing.assert_array_equal(loaded.select_action(state, True), agent.select_action(state, True))
    wrong = CSBAPRAgent(4, 2, CSBAPRConfig(hidden_dim=8, actor_type=None, device="cpu"))
    with pytest.raises(ValueError, match="actor type"):
        wrong.load(path)
    with pytest.raises(ValueError, match="action-valued reference_builder"):
        CSBAPRAgent(4, 2, cfg).sindy_preidentify(env)
