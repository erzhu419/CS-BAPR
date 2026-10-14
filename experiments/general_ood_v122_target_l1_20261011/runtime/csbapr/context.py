"""
Online transition context for non-stationary/OOD continuous-control tasks.

The bus setting exposes a meaningful regime structure through demand and
headway dynamics. Generic Gym/MuJoCo tasks do not. This tracker gives the
policy and critic a compact, online descriptor of the current regime using
only recent transition statistics, without oracle access to the OOD profile.
"""

from __future__ import annotations

import numpy as np


class TransitionContextTracker:
    """
    Deterministic latent-context estimator from recent transitions.

    The raw signal is [delta_s, |delta_s|, action, reward, ||delta_s||,
    ||action||]. We normalize it against long-run online statistics and keep
    an episode-local EMA, so abrupt regime shifts produce a non-zero context
    while stationary behavior decays back toward zero.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ema_alpha: float = 0.2,
        clip: float = 5.0,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.context_dim = 2 * self.state_dim + self.action_dim + 3
        self.ema_alpha = float(ema_alpha)
        self.clip = float(clip)

        self._count = 0
        self._mean = np.zeros(self.context_dim, dtype=np.float32)
        self._m2 = np.zeros(self.context_dim, dtype=np.float32)
        self._ema = np.zeros(self.context_dim, dtype=np.float32)

    def reset_episode(self):
        """Reset short-term context while preserving long-run normalizers."""
        self._ema.fill(0.0)

    @property
    def context(self) -> np.ndarray:
        return self._ema.astype(np.float32, copy=True)

    def update(self, state, action, reward, next_state, done=False) -> np.ndarray:
        raw = self._raw_features(state, action, reward, next_state)
        self._update_running_stats(raw)

        std = np.sqrt(self._variance() + 1e-6)
        z = np.clip((raw - self._mean) / std, -self.clip, self.clip)
        z = np.tanh(z / self.clip).astype(np.float32)

        self._ema = (
            (1.0 - self.ema_alpha) * self._ema + self.ema_alpha * z
        ).astype(np.float32)
        if done:
            # Keep the terminal transition signal for the stored next_context,
            # but callers should invoke reset_episode() before the next reset.
            pass
        return self.context

    def _raw_features(self, state, action, reward, next_state) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if state.shape[0] != self.state_dim:
            raise ValueError(f"state_dim mismatch: expected {self.state_dim}, got {state.shape[0]}")
        if next_state.shape[0] != self.state_dim:
            raise ValueError(f"next_state_dim mismatch: expected {self.state_dim}, got {next_state.shape[0]}")
        if action.shape[0] != self.action_dim:
            raise ValueError(f"action_dim mismatch: expected {self.action_dim}, got {action.shape[0]}")

        delta = next_state - state
        scalars = np.asarray(
            [float(reward), float(np.linalg.norm(delta)), float(np.linalg.norm(action))],
            dtype=np.float32,
        )
        return np.concatenate([delta, np.abs(delta), action, scalars]).astype(np.float32)

    def _update_running_stats(self, raw: np.ndarray):
        self._count += 1
        if self._count == 1:
            self._mean = raw.astype(np.float32, copy=True)
            self._m2.fill(0.0)
            return
        delta = raw - self._mean
        self._mean = self._mean + delta / float(self._count)
        delta2 = raw - self._mean
        self._m2 = self._m2 + delta * delta2

    def _variance(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(self.context_dim, dtype=np.float32)
        return np.maximum(self._m2 / float(self._count - 1), 1e-6).astype(np.float32)
