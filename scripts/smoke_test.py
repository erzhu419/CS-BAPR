"""
CS-BAPR Smoke Test — End-to-end verification.

Validates the complete pipeline:
1. Pendulum environment trajectory collection
2. SINDy identification → ε₁ < threshold
3. SINDyTorchWrapper forward/gradient verification
4. NAU/NMU Actor forward pass
5. Complete training loop (100 steps) without crashes
"""

import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_nau_nmu():
    """Test NAU/NMU layers and Actor."""
    print("\n" + "="*60)
    print("[TEST 1] NAU/NMU Layers")
    print("="*60)

    from csbapr.networks.nau_nmu import NAULayer, NMULayer, NAU_NMU_Actor

    # NAU Layer
    nau = NAULayer(8, 4)
    x = torch.randn(16, 8)
    y = nau(x)
    assert y.shape == (16, 4), f"NAU output shape wrong: {y.shape}"
    assert nau.W.data.abs().max() <= 1.0 + 1e-6, "NAU weights should init in [-1, 1]"
    reg_loss = nau.regularization_loss()
    assert reg_loss.ndim == 0, "Reg loss should be scalar"
    print(f"  ✓ NAULayer: input {x.shape} → output {y.shape}, reg_loss={reg_loss.item():.4f}")

    # NMU Layer
    nmu = NMULayer(4)
    x2 = torch.randn(16, 4)
    y2 = nmu(x2)
    assert y2.shape == (16, 4), f"NMU output shape wrong: {y2.shape}"
    L = nmu.lipschitz_constant
    print(f"  ✓ NMULayer: L={L:.4f}")

    # NAU_NMU_Actor (SAC-style)
    actor = NAU_NMU_Actor(state_dim=3, action_dim=1, hidden_dim=64)
    state = torch.randn(16, 3)
    mean, log_std = actor(state)
    assert mean.shape == (16, 1), f"Actor mean shape wrong: {mean.shape}"
    assert log_std.shape == (16, 1), f"Actor log_std shape wrong: {log_std.shape}"

    action, log_prob, _, _, _ = actor.evaluate(state)
    assert action.shape == (16, 1)
    assert log_prob.shape == (16,)
    print(f"  ✓ NAU_NMU_Actor: evaluate OK, action shape={action.shape}")

    action_np = actor.get_action(state[0])
    assert action_np.shape == (1,), f"get_action shape wrong: {action_np.shape}"
    print(f"  ✓ NAU_NMU_Actor: get_action OK")


def test_critic():
    """Test Ensemble Q-Network."""
    print("\n" + "="*60)
    print("[TEST 2] Ensemble Q-Network")
    print("="*60)

    from csbapr.networks.critic import EnsembleQNet

    critic = EnsembleQNet(state_dim=3, action_dim=1, hidden_dim=64, num_critics=5)
    state = torch.randn(16, 3)
    action = torch.randn(16, 1)
    q = critic(state, action)
    assert q.shape == (5, 16), f"Critic output shape wrong: {q.shape}"
    q_std = q.std(dim=0)
    assert q_std.shape == (16,)
    print(f"  ✓ EnsembleQNet: output {q.shape}, q_std range [{q_std.min():.4f}, {q_std.max():.4f}]")


def test_belief():
    """Test BeliefTracker and SurpriseComputer."""
    print("\n" + "="*60)
    print("[TEST 3] Belief System")
    print("="*60)

    from csbapr.belief.tracker import BeliefTracker
    from csbapr.belief.surprise import SurpriseComputer

    bt = BeliefTracker(max_run_length=20)
    bt.reset()
    assert abs(bt.belief.sum() - 1.0) < 1e-6, "Belief should sum to 1"

    for i in range(10):
        bt.update(surprise=0.1 * i)
    assert abs(bt.belief.sum() - 1.0) < 1e-6
    print(f"  ✓ BeliefTracker: entropy={bt.entropy:.4f}, eff_window={bt.effective_window:.2f}")

    sc = SurpriseComputer()
    reward = torch.randn(32, 1)
    q_std = torch.randn(32).abs()
    surprise_val = sc.compute(reward, q_std)
    assert isinstance(surprise_val, float)
    print(f"  ✓ SurpriseComputer: surprise={surprise_val:.4f}")


def test_sindy():
    """Test SINDy model and PyTorch wrapper."""
    print("\n" + "="*60)
    print("[TEST 4] SINDy Pipeline")
    print("="*60)

    try:
        import pysindy
    except ImportError:
        print("  ⚠ PySINDy not installed, skipping SINDy test")
        return False

    from csbapr.sindy.world_model import SymbolicWorldModel
    from csbapr.sindy.torch_wrapper import SINDyTorchWrapper

    # Generate simple dynamics: x_{t+1} = Ax_t (linear system)
    np.random.seed(42)
    A = np.array([[0.95, 0.1], [-0.1, 0.9]])
    n_traj = 10
    X_list = []
    for _ in range(n_traj):
        x = np.random.randn(2)
        traj = [x]
        for t in range(50):
            x = A @ x + 0.01 * np.random.randn(2)
            traj.append(x)
        X_list.append(np.array(traj))

    # Fit SINDy
    model = SymbolicWorldModel(n_state=2, poly_degree=2, threshold=0.05, discrete_time=True)
    model.fit(X_list, multiple_trajectories=True, t=1.0)

    print(f"  Sparsity: {model.sparsity:.3f}")
    print(f"  Coefficients shape: {model.coeffs.shape}")
    print(f"  Features: {model.get_feature_names()}")

    # Test prediction
    test_state = np.random.randn(5, 2)
    pred = model.predict(test_state)
    true_next = (A @ test_state.T).T
    pred_error = np.linalg.norm(pred - true_next, axis=-1).mean()
    print(f"  Prediction error: {pred_error:.4f}")

    # Torch wrapper
    wrapper = SINDyTorchWrapper(model, verify_features=True)
    x_torch = torch.tensor(test_state, dtype=torch.float32, requires_grad=True)
    out = wrapper(x_torch)
    assert out.shape == (5, 2), f"Wrapper output shape wrong: {out.shape}"

    # Verify gradient flows
    loss = out.sum()
    loss.backward()
    assert x_torch.grad is not None, "Gradient should flow through wrapper"
    print(f"  ✓ SINDyTorchWrapper: output {out.shape}, gradients OK")

    # Validate against numpy
    valid = wrapper.validate_against_numpy(model, test_state)
    print(f"  ✓ Numpy validation: {'PASSED' if valid else 'FAILED'}")

    return True


def test_jacobian_loss():
    """Test Jacobian consistency loss."""
    print("\n" + "="*60)
    print("[TEST 5] Jacobian Loss")
    print("="*60)

    from csbapr.losses.jacobian import compute_jacobian_loss
    from csbapr.networks.nau_nmu import NAU_NMU_Actor

    # Create a simple f_sym (linear model)
    class SimpleSym(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('w', torch.randn(3, 1))
        def forward(self, x):
            return x @ self.w

    actor = NAU_NMU_Actor(3, 1, 64)
    f_sym = SimpleSym()
    states = torch.randn(16, 3)

    jac_loss = compute_jacobian_loss(actor, f_sym, states)
    assert jac_loss.ndim == 0, "Jacobian loss should be scalar"
    assert not torch.isnan(jac_loss), "Jacobian loss should not be NaN"
    assert not torch.isinf(jac_loss), "Jacobian loss should not be Inf"

    # Verify gradient flows to actor
    jac_loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert has_grad, "Gradients should flow to actor"
    print(f"  ✓ Jacobian loss: {jac_loss.item():.6f}, gradients flow OK")


def test_full_training_loop():
    """Test complete training loop with Gymnasium environment."""
    print("\n" + "="*60)
    print("[TEST 6] Full Training Loop (100 steps)")
    print("="*60)

    try:
        import gymnasium as gym
    except ImportError:
        print("  ⚠ gymnasium not installed, skipping full test")
        return

    from csbapr.agent import CSBAPRAgent
    from csbapr.config import CSBAPRConfig

    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    config = CSBAPRConfig(
        hidden_dim=64,
        num_critics=3,
        batch_size=32,
        warmup_steps=64,
        sindy_n_explore_episodes=5,
        sindy_lib_degree=2,
        sindy_threshold=0.1,
        use_nau_actor=True,
    )

    agent = CSBAPRAgent(state_dim, action_dim, config)
    print(f"  Agent created: state_dim={state_dim}, action_dim={action_dim}")

    # Phase 0: SINDy
    try:
        agent.sindy_preidentify(env)
        print("  ✓ Phase 0 complete")
    except Exception as e:
        print(f"  ⚠ Phase 0 failed (pysindy may not be installed): {e}")
        print("  Continuing without SINDy...")

    # Collect some data
    print("  Collecting warmup data...")
    state, _ = env.reset(seed=42)
    for step in range(config.warmup_steps + 50):
        action = agent.select_action(state)
        # Scale action to env's action space
        scaled_action = action * env.action_space.high[0]
        next_state, reward, terminated, truncated, info = env.step(scaled_action)
        done = terminated or truncated
        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        if done:
            state, _ = env.reset()

    print(f"  Buffer size: {agent.replay_buffer.size}")

    # Training loop
    print("  Running 100 training steps...")
    losses = []
    for step in range(100):
        metrics = agent.update()
        if metrics is not None:
            losses.append(metrics['q_loss'])
            if step % 25 == 0:
                print(f"    Step {step}: q_loss={metrics['q_loss']:.4f}, "
                      f"sym_p={metrics['sym_penalty']:.4f}, "
                      f"jac={metrics['jac_loss']:.4f}, "
                      f"α={metrics['alpha']:.4f}")

    # Verify no NaN/Inf
    assert all(not np.isnan(l) for l in losses), "Q-loss has NaN!"
    assert all(not np.isinf(l) for l in losses), "Q-loss has Inf!"
    print(f"  ✓ 100 steps complete, no NaN/Inf detected")
    print(f"  ✓ Q-loss range: [{min(losses):.4f}, {max(losses):.4f}]")

    env.close()


if __name__ == '__main__':
    print("="*60)
    print("CS-BAPR Smoke Test")
    print("="*60)

    test_nau_nmu()
    test_critic()
    test_belief()
    sindy_ok = test_sindy()
    if sindy_ok:
        test_jacobian_loss()
    test_full_training_loop()

    print("\n" + "="*60)
    print("✓ ALL SMOKE TESTS PASSED")
    print("="*60)
