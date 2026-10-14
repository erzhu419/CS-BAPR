"""Double-integrator regulation with observed, persistent random external force."""

import numpy as np
from gym.spaces.box import Box

from csbapr.envs.lqr_env import A, B, Q, R, ACTION_SCALE, EP_LEN, LQREnv

INITIAL_AMPLITUDE = 0.2
BASE_FORCE_CAP = 0.05
REFRESH_PROBABILITY = 0.1
BASE_STORM_PROBABILITY = 0.05
TAIL_SHAPE = 1.5
SEVERITIES = (1, 2, 5, 10, 20, 50)


class RandomLoadEnv(LQREnv):
    """Observation: position, velocity, and the external force applied now.

    The force process is action-independent. A constant refresh hazard gives
    mean dwell time ten steps; no future force or severity label is observed.
    """

    def __init__(self, severity=1.0, seed=None):
        if severity not in SEVERITIES:
            raise ValueError(f"Supported severities are {SEVERITIES}")
        super().__init__(amplitude=INITIAL_AMPLITUDE, seed=seed)
        self.severity = float(severity)
        self.force_cap = BASE_FORCE_CAP * self.severity
        self.storm_probability = BASE_STORM_PROBABILITY * np.sqrt(self.severity)
        self.observation_space = Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.disturbance = np.zeros(2)
        self.is_storm = False

    @staticmethod
    def specification():
        return {
            "initial_amplitude": INITIAL_AMPLITUDE,
            "base_force_cap": BASE_FORCE_CAP,
            "refresh_probability": REFRESH_PROBABILITY,
            "base_storm_probability": BASE_STORM_PROBABILITY,
            "storm_probability_rule": "base_probability * sqrt(severity)",
            "tail_shape": TAIL_SHAPE,
            "tail_rule": "min(1, 0.1 * (1 + Pareto(tail_shape)))",
            "background_rule": "Uniform(-0.1, 0.1) * force_cap",
            "action_force_scale": ACTION_SCALE,
            "episode_length": EP_LEN,
            "observation": ["x", "y", "vx", "vy", "current_force_x", "current_force_y"],
        }

    def _draw_disturbance(self):
        # Fixed draws couple severities without branching the random stream.
        self.is_storm = bool(self._rng.random() < self.storm_probability)
        background = self._rng.uniform(-0.1, 0.1, 2)
        signs = self._rng.choice([-1.0, 1.0], 2)
        tail = np.minimum(1.0, 0.1 * (1.0 + self._rng.pareto(TAIL_SHAPE, 2)))
        self.disturbance = self.force_cap * (signs * tail if self.is_storm else background)

    def _observation(self):
        return np.concatenate((self.state, self.disturbance)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._draw_disturbance()
        return self._observation(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        controlled_force = ACTION_SCALE * action
        applied_disturbance = self.disturbance.copy()
        storm = self.is_storm
        cost = float(self.state @ Q @ self.state + controlled_force @ R @ controlled_force)
        self.state = A @ self.state + B @ (controlled_force + applied_disturbance)
        self.t += 1
        done = self.t >= EP_LEN
        if not done and self._rng.random() < REFRESH_PROBABILITY:
            self._draw_disturbance()
        return self._observation(), -cost, done, False, {
            "cost": cost, "applied_disturbance": applied_disturbance,
            "storm": storm,
        }
