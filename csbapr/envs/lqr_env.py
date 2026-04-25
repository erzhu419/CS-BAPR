"""
LQR-v0: 2D point-mass double integrator under LQR-style cost.

Designed as a positive case for the SINDy / NAU framework:
    - Dynamics are linear: x_{t+1} = A x_t + B u_t (SINDy identifies A,B exactly)
    - Optimal policy is linear: u* = -K x (NAU's L=0 derivative is exact)
    - OOD axis: initial-state amplitude m ∈ {1, 2, 5, 10, 20, 50}
      (training samples x0 in unit ball; OOD samples x0 in radius-m ball)

State (4D): [x, y, vx, vy]
Action (2D): [Fx, Fy] in [-1, 1] (clipped)
Reward: -(x' Q x + u' R u), with Q=diag(1,1,0.1,0.1), R=diag(0.01, 0.01)
Episode: T=200 steps, dt=0.05
"""
from __future__ import annotations
import numpy as np
import gym
from gym.spaces.box import Box


# ── Continuous-time dynamics ──
# m \ddot x = u, m \ddot y = u_y, with mass m=1
# Discrete-time (dt=0.05):
#   x_{t+1} = x_t + v_t dt + 0.5 a dt^2
#   v_{t+1} = v_t + a dt

DT = 0.05
A = np.array([
    [1.0, 0.0, DT,  0.0],
    [0.0, 1.0, 0.0, DT],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
B = np.array([
    [0.5 * DT * DT, 0.0],
    [0.0, 0.5 * DT * DT],
    [DT, 0.0],
    [0.0, DT],
], dtype=np.float64)

Q = np.diag([1.0, 1.0, 0.1, 0.1])
R = np.diag([0.01, 0.01])

ACTION_SCALE = 5.0   # action in [-1,1] ⇒ force in [-5, 5]

EP_LEN = 200


def solve_lqr_K(A, B, Q, R, n_iter=500):
    """Discrete algebraic Riccati equation by iteration → optimal gain K."""
    P = Q.copy()
    for _ in range(n_iter):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
        if np.linalg.norm(P_next - P) < 1e-10:
            break
        P = P_next
    return K  # shape (2, 4)


# Pre-compute the optimal feedback gain (used only for diagnostics)
K_OPT = solve_lqr_K(A, B * ACTION_SCALE, Q, R)


class LQREnv(gym.Env):
    """
    Linear-quadratic regulator on a 2D double integrator.

    Args:
        amplitude: scale factor on the initial-state distribution.
            amplitude=1 → x0 uniformly in the unit ball (training default).
            amplitude>1 → OOD: x0 uniformly in a ball of radius `amplitude`.
            This is the analog of the bus env's `od_mult`.
        seed: numpy seed.
    """

    metadata = {"render_modes": []}

    def __init__(self, amplitude: float = 1.0, seed: int | None = None):
        super().__init__()
        self.amplitude = float(amplitude)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float64)
        self.t = 0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Sample x0 uniformly in radius-1 ball, scale by amplitude
        # 4D ball-uniform via gaussian-then-normalize trick
        v = self._rng.standard_normal(4)
        v = v / np.linalg.norm(v)
        r = self._rng.uniform(0, 1) ** (1.0 / 4)
        self.state = self.amplitude * r * v
        self.t = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        force = ACTION_SCALE * action  # scale [-1,1] → [-5, 5]

        # Cost: x' Q x + u' R u
        cost = float(self.state @ Q @ self.state + force @ R @ force)
        reward = -cost

        # Linear discrete dynamics
        self.state = A @ self.state + B @ force

        self.t += 1
        terminated = bool(self.t >= EP_LEN)
        truncated = False
        # Auto-terminate if state explodes (safety; shouldn't happen with good policy)
        if np.linalg.norm(self.state) > 1e4:
            terminated = True

        info = {"cost": cost}
        return self.state.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        pass


def get_dynamics_matrices():
    """Return (A, B) of the continuous-/discrete-time linear system."""
    return A.copy(), B.copy()


def get_optimal_gain():
    """Return the LQR optimal gain K (action = -K @ state, before clipping)."""
    return K_OPT.copy()
