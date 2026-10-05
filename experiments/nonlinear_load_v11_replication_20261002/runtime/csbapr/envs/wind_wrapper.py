"""
WindDisturbanceWrapper: continuous-amplitude external force OOD for MuJoCo.

Standard MuJoCo OOD profiles (mass=10x, gravity=5x) cause physically
degenerate behaviour (robot can't move) rather than a graceful continuous
degradation. The bus benchmark's success comes from a *continuous, gentle*
OOD axis (od_mult ∈ [1, 50]) along which the policy's response can be
plotted and bounded.

This wrapper supplies an analogous axis on MuJoCo: a horizontal "wind"
disturbance force on a target body, with amplitude ∈ {0, 1, 2, 5, 10, 20}
in units of body-mass × gravity. amp=0 reproduces the unperturbed env;
amp=20 is severe but the underlying physics remains well-defined.

Train: amp=0 (no wind).
OOD eval: sweep amp ∈ {1, 2, 5, 10, 20}.

Usage:
    env = gym.make("Hopper-v4")
    env = WindDisturbanceWrapper(env, body_name="torso")
    env.set_wind_amplitude(5.0)   # in units of m_body * g
    obs, _ = env.reset()
    obs, r, term, trunc, info = env.step(action)
"""
from __future__ import annotations
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


def _resolve_body_id(model, body_name: str) -> int:
    """Find the MuJoCo body id by name. Handles both old- and new-style APIs."""
    # MuJoCo 2.3+ API
    if hasattr(model, "body"):
        try:
            return model.body(body_name).id
        except Exception:
            pass
    # Older fallback: search body_names array
    for i in range(model.nbody):
        try:
            name = model.body(i).name
        except Exception:
            name = None
        if name == body_name:
            return i
    raise ValueError(f"body '{body_name}' not found in model")


class WindDisturbanceWrapper(gym.Wrapper):
    """
    Adds a constant horizontal force on a target body for the duration of
    the episode. Force = amplitude × body_mass × |gravity|.

    Args:
        env: a gymnasium MuJoCo env (Hopper-v4, HalfCheetah-v4, Walker2d-v4, …).
        body_name: which body to push. Defaults to 'torso' (works for the
            three locomotion tasks above).
        direction: 3-vector force direction in world coords (default = +x).
    """

    def __init__(
        self,
        env,
        body_name: str = "torso",
        direction=(1.0, 0.0, 0.0),
        amplitude: float = 0.0,
    ):
        super().__init__(env)
        self.body_name = body_name
        self.direction = np.asarray(direction, dtype=np.float64)
        self.direction = self.direction / max(np.linalg.norm(self.direction), 1e-9)
        self.amplitude = float(amplitude)
        self._body_id = None
        self._body_mass = None
        self._g_norm = None
        self._cached_force = np.zeros(3, dtype=np.float64)

    def set_wind_amplitude(self, amplitude: float):
        """Set the wind force amplitude (in units of body_mass × g)."""
        self.amplitude = float(amplitude)
        self._cached_force = self._compute_force()

    def _compute_force(self) -> np.ndarray:
        if (
            self._body_id is None
            or self._body_mass is None
            or self._g_norm is None
            or self.amplitude == 0.0
        ):
            return np.zeros(3, dtype=np.float64)
        return self.amplitude * self._body_mass * self._g_norm * self.direction

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
        else:  # old gym
            obs, info = out, {}

        # Resolve body id and look up mass / gravity, lazily on first reset
        if self._body_id is None:
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "model"):
                model = unwrapped.model
                try:
                    self._body_id = _resolve_body_id(model, self.body_name)
                    self._body_mass = float(model.body_mass[self._body_id])
                    self._g_norm = float(np.linalg.norm(model.opt.gravity))
                except Exception as e:
                    print(f"[WindWrapper] body resolution failed: {e!r}; wind disabled.")
                    self._body_id = -1
            else:
                # Non-MuJoCo env: wind is a no-op
                self._body_id = -1
        self._cached_force = self._compute_force()
        return obs, info

    def step(self, action):
        # Apply wind via xfrc_applied (per-body world-frame external wrench)
        if (
            self._body_id is not None
            and self._body_id >= 0
            and self.amplitude != 0.0
        ):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "data"):
                data = unwrapped.data
                # xfrc_applied shape: [nbody, 6] = [Fx, Fy, Fz, Tx, Ty, Tz]
                # Set then call physics step (the env will do that).
                data.xfrc_applied[self._body_id, :3] = self._cached_force
        return self.env.step(action)
