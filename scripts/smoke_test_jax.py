"""CS-BAPR JAX Smoke Test.

Tests the JAX implementation WITHOUT Brax (pure JAX/Flax NNX tests).
Brax environment tests require the jax-rl conda env.

Tests:
1. NAU/NMU Policy: forward, sample, deterministic, regularization
2. SINDy JAX Wrapper: polynomial features, predict, gradient flow
3. Jacobian consistency loss: vmap+jacrev, stop_gradient on f_sym
4. CS-BAPR Agent: init, scan-fused update (mock data)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


def test_nau_nmu_policy():
    """Test NAU/NMU Policy network."""
    print("\n" + "="*60)
    print("[TEST 1] NAU/NMU Policy (JAX)")
    print("="*60)

    from csbapr_jax.networks.nau_nmu import NAULayer, NMULayer, NAU_NMU_Policy

    rngs = nnx.Rngs(42)

    # NAU Layer
    nau = NAULayer(8, 4, rngs=rngs)
    x = jnp.ones((16, 8))
    y = nau(x)
    assert y.shape == (16, 4), f"NAU shape wrong: {y.shape}"
    reg = nau.regularization_loss()
    assert reg.ndim == 0
    print(f"  ✓ NAULayer: {x.shape} → {y.shape}, reg={float(reg):.4f}")

    # NMU Layer
    nmu = NMULayer(4, rngs=rngs)
    x2 = jnp.ones((16, 4)) * 2.0
    y2 = nmu(x2)
    assert y2.shape == (16, 4)
    L = float(nmu.lipschitz_constant)
    print(f"  ✓ NMULayer: L={L:.4f}")

    # NAU_NMU_Policy
    policy = NAU_NMU_Policy(obs_dim=11, act_dim=3, hidden_dim=64,
                            ep_dim=2, n_layers=2, rngs=rngs)
    obs = jnp.ones((16, 11))
    ep = jnp.ones((16, 2))

    mean, log_std = policy(obs, ep)
    assert mean.shape == (16, 3), f"Mean shape: {mean.shape}"
    assert log_std.shape == (16, 3)
    print(f"  ✓ Forward: mean={mean.shape}, log_std={log_std.shape}")

    key = jax.random.PRNGKey(0)
    action, log_prob = policy.sample(obs, key, ep)
    assert action.shape == (16, 3)
    assert log_prob.shape == (16,)
    assert jnp.all(jnp.abs(action) <= 1.0)  # tanh squashed
    print(f"  ✓ Sample: action={action.shape}, log_prob={log_prob.shape}")

    det_act = policy.deterministic(obs, ep)
    assert det_act.shape == (16, 3)
    print(f"  ✓ Deterministic: {det_act.shape}")

    # Without context (ep_tensor=None)
    policy_no_ep = NAU_NMU_Policy(obs_dim=11, act_dim=3, hidden_dim=64,
                                   ep_dim=0, n_layers=2, rngs=nnx.Rngs(99))
    mean2, _ = policy_no_ep(obs)
    assert mean2.shape == (16, 3)
    print(f"  ✓ Without context: OK")


def test_sindy_jax_wrapper():
    """Test SINDy JAX wrapper."""
    print("\n" + "="*60)
    print("[TEST 2] SINDy JAX Wrapper")
    print("="*60)

    from csbapr_jax.sindy.jax_wrapper import SINDyJAXWrapper

    # Create mock SINDy coefficients for a 3D system with degree=2
    # Features: [1, x0, x1, x2, x0², x0x1, x0x2, x1², x1x2, x2²] = 10 features
    n_state = 3
    n_features = 10  # 1 + 3 + 6 for degree=2
    coeffs = np.random.randn(n_state, n_features).astype(np.float32) * 0.1

    wrapper = SINDyJAXWrapper(coeffs, n_state=n_state, poly_degree=2)

    # Test predict
    x = jnp.ones((16, 3))
    pred = wrapper.predict(x)
    assert pred.shape == (16, 3), f"Predict shape: {pred.shape}"
    print(f"  ✓ Predict: {pred.shape}")

    # Test single predict
    pred_single = wrapper.predict_single(x[0])
    assert pred_single.shape == (3,)
    assert jnp.allclose(pred_single, pred[0], atol=1e-5)
    print(f"  ✓ Single predict matches batch: OK")

    # Test gradient flows
    grad_fn = jax.grad(lambda x: wrapper.predict_single(x).sum())
    g = grad_fn(x[0])
    assert g.shape == (3,)
    assert not jnp.any(jnp.isnan(g))
    print(f"  ✓ Gradient: shape={g.shape}, no NaN")


def test_jacobian_loss():
    """Test Jacobian consistency loss."""
    print("\n" + "="*60)
    print("[TEST 3] Jacobian Consistency Loss (JAX)")
    print("="*60)

    from csbapr_jax.sindy.jax_wrapper import SINDyJAXWrapper, compute_jacobian_loss_jax
    from csbapr_jax.networks.nau_nmu import NAU_NMU_Policy

    rngs = nnx.Rngs(42)
    policy = NAU_NMU_Policy(obs_dim=3, act_dim=1, hidden_dim=32,
                            ep_dim=0, n_layers=2, rngs=rngs)

    # Mock SINDy: 3 state dims, degree=2 → 10 features, 3 outputs
    # But action dim=1, so SINDy output gets trimmed to 1
    n_features = 10
    coeffs = np.random.randn(3, n_features).astype(np.float32) * 0.1
    wrapper = SINDyJAXWrapper(coeffs, n_state=3, poly_degree=2)

    obs = jnp.ones((8, 3))
    key = jax.random.PRNGKey(0)

    def policy_fn(x):
        return policy(x)

    jac_loss = compute_jacobian_loss_jax(policy_fn, wrapper, obs, key)
    assert jac_loss.ndim == 0, f"Jac loss should be scalar, got ndim={jac_loss.ndim}"
    assert not jnp.isnan(jac_loss)
    assert not jnp.isinf(jac_loss)
    print(f"  ✓ Jacobian loss: {float(jac_loss):.6f}")

    # Verify gradient flows to policy params
    p_params = nnx.state(policy, nnx.Param)
    gd_policy = nnx.graphdef(policy)

    def loss_fn(pp):
        pm = nnx.merge(gd_policy, pp)
        def pfn(x):
            return pm(x)
        return compute_jacobian_loss_jax(pfn, wrapper, obs, key)

    grads = jax.grad(loss_fn)(p_params)
    has_nonzero = False
    for leaf in jax.tree.leaves(grads):
        if jnp.abs(leaf).sum() > 0:
            has_nonzero = True
            break
    assert has_nonzero, "Gradients should flow to policy"
    print(f"  ✓ Gradients flow to policy: OK")


def test_csbapr_agent():
    """Test CS-BAPR agent initialization and mock update."""
    print("\n" + "="*60)
    print("[TEST 4] CS-BAPR Agent (mock data)")
    print("="*60)

    from csbapr_jax.configs.default import CSBAPRConfig
    from csbapr_jax.algos.csbapr import CSBAPR

    config = CSBAPRConfig(
        hidden_dim=64,
        ensemble_size=3,
        ep_dim=2,
        use_nau_actor=True,
        lr=3e-4,
    )

    obs_dim = 11
    act_dim = 3
    agent = CSBAPR(obs_dim, act_dim, config, seed=42)
    print(f"  ✓ Agent created: obs_dim={obs_dim}, act_dim={act_dim}")

    # Test select_action
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.select_action(obs)
    assert action.shape == (act_dim,), f"Action shape: {action.shape}"
    print(f"  ✓ select_action: {action.shape}")

    det_action = agent.select_action(obs, deterministic=True)
    assert det_action.shape == (act_dim,)
    print(f"  ✓ select_action(det): {det_action.shape}")

    # Install mock SINDy
    # For degree=2, n_state=11: n_features = 1 + 11 + C(12,2) = 1 + 11 + 66 = 78
    n_features = 1 + obs_dim + obs_dim * (obs_dim + 1) // 2  # = 78
    coeffs = np.random.randn(obs_dim, n_features).astype(np.float32) * 0.01
    agent.set_sindy_coeffs(coeffs, n_state=obs_dim, poly_degree=2)
    print(f"  ✓ SINDy coefficients installed ({obs_dim}x{n_features})")

    # Create mock stacked batch (simulating replay_buffer.sample_stacked)
    n_updates = 5
    batch_size = 32
    stacked = {
        "obs": np.random.randn(n_updates, batch_size, obs_dim).astype(np.float32),
        "act": np.random.randn(n_updates, batch_size, act_dim).astype(np.float32),
        "rew": np.random.randn(n_updates, batch_size, 1).astype(np.float32),
        "next_obs": np.random.randn(n_updates, batch_size, obs_dim).astype(np.float32),
        "done": np.zeros((n_updates, batch_size, 1), dtype=np.float32),
        "task_id": np.zeros((n_updates, batch_size), dtype=np.int32),
    }

    print(f"  Running multi_update (first call compiles, may take a moment)...")
    metrics = agent.multi_update(stacked, current_iter=0)

    assert "critic_loss" in metrics
    assert "policy_loss" in metrics
    assert "jac_loss" in metrics
    assert "weighted_lambda" in metrics
    assert not np.isnan(metrics["critic_loss"])
    assert not np.isnan(metrics["policy_loss"])

    print(f"  ✓ multi_update: critic_loss={metrics['critic_loss']:.4f}, "
          f"policy_loss={metrics['policy_loss']:.4f}, "
          f"jac_loss={metrics['jac_loss']:.4f}, "
          f"α={metrics['alpha']:.4f}")

    # Second update (should use cached JIT)
    metrics2 = agent.multi_update(stacked, current_iter=1)
    print(f"  ✓ 2nd update: critic_loss={metrics2['critic_loss']:.4f}, "
          f"jac_loss={metrics2['jac_loss']:.4f}")


if __name__ == '__main__':
    print("="*60)
    print("CS-BAPR JAX Smoke Test")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    test_nau_nmu_policy()
    test_sindy_jax_wrapper()
    test_jacobian_loss()
    test_csbapr_agent()

    print("\n" + "="*60)
    print("✓ ALL JAX SMOKE TESTS PASSED")
    print("="*60)
