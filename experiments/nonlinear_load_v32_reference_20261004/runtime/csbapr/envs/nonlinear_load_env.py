"""Observed random load with a fixed monotone nonlinear actuator."""

import numpy as np

from csbapr.envs.lqr_env import A, B, Q, R, ACTION_SCALE, EP_LEN
from csbapr.envs.random_load_env import RandomLoadEnv, REFRESH_PROBABILITY

LINEAR_SHARE = 0.5


def actuator_force(action):
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return ACTION_SCALE * (LINEAR_SHARE * action + (1 - LINEAR_SHARE) * action ** 3)


def inverse_force(force):
    from scipy.optimize import brentq

    return np.array([brentq(lambda u: actuator_force(u) - f, -1.0, 1.0) for f in force])


def local_linear_gain():
    from scipy.linalg import solve_discrete_are

    control = B * ACTION_SCALE * LINEAR_SHARE
    action_cost = R * ACTION_SCALE ** 2
    p = solve_discrete_are(A, control, Q, action_cost)
    gain = np.linalg.solve(action_cost + control.T @ p @ control, control.T @ p @ A)
    return np.hstack((gain, np.eye(2) / (ACTION_SCALE * LINEAR_SHARE)))


class NonlinearLoadEnv(RandomLoadEnv):
    """Same load/reset process; cost measures command effort, not output force."""

    @staticmethod
    def specification():
        return {
            **RandomLoadEnv.specification(),
            "environment": "NonlinearLoad-v0",
            "actuator_rule": "5 * (0.5 * u + 0.5 * u^3), componentwise",
            "linear_share": LINEAR_SHARE,
            "local_force_gain": ACTION_SCALE * LINEAR_SHARE,
            "maximum_force": ACTION_SCALE,
            "cost_rule": "s_physical^T Q s_physical + (5*u)^T R (5*u)",
            "state_cost_diagonal": np.diag(Q).tolist(),
            "command_force_cost_diagonal": np.diag(R).tolist(),
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        command_force = ACTION_SCALE * action
        output_force = actuator_force(action)
        applied_disturbance, storm = self.disturbance.copy(), self.is_storm
        cost = float(self.state @ Q @ self.state + command_force @ R @ command_force)
        self.state = A @ self.state + B @ (output_force + applied_disturbance)
        self.t += 1
        done = self.t >= EP_LEN
        if not done and self._rng.random() < REFRESH_PROBABILITY:
            self._draw_disturbance()
        return self._observation(), -cost, done, False, {
            "cost": cost, "applied_disturbance": applied_disturbance, "storm": storm,
            "command_force": command_force, "actuator_force": output_force,
        }
