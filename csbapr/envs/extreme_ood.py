"""
General extreme-OOD wrappers for Gymnasium environments.

The bus benchmark stresses policies with very large demand multipliers and
mid-episode regime changes. This module provides the same style of stressor
for ordinary RL environments:

1. If the environment exposes mutable physics parameters, perturb them.
2. Otherwise, fall back to observation/action/reward perturbations.
3. Support static, abrupt, oscillating, per-episode, and step-random schedules.

The wrapper is intentionally environment-agnostic and does not depend on
CS-BAPR internals, so it can be used for baselines as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility fallback
    import gym


Profile = Dict[str, Any]


@dataclass
class ExtremeOODConfig:
    """Configuration for the general OOD schedule."""

    intensity: float = 50.0
    schedule: str = "episode_random"
    profile_family: str = "extreme"
    seed: Optional[int] = None
    static_profile: Optional[str] = None
    abrupt_step: int = 100
    switch_every_steps: int = 100
    switch_probability: float = 0.35
    include_sensor_noise: bool = True
    include_action_noise: bool = True
    include_reward_noise: bool = False


def _finite_box_clip(space, value):
    if not hasattr(space, "low") or not hasattr(space, "high"):
        return value
    low = np.asarray(space.low)
    high = np.asarray(space.high)
    if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
        return np.clip(value, low, high)
    return value


def _safe_float(value, default=1.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    if not np.isfinite(value):
        return default
    return value


def default_extreme_profiles(env_name: Optional[str] = None, intensity: float = 50.0) -> List[Profile]:
    """
    Build bus-like extreme OOD profiles.

    Multipliers are deliberately much larger than ordinary domain
    randomization. The physical parameters are used when supported; the
    sensor/action keys are used everywhere.
    """
    hi = max(float(intensity), 1.0)
    mid = float(np.sqrt(hi))
    lo = 1.0 / hi
    env_name = env_name or ""

    common = [
        {"name": "id", "obs_scale": 1.0, "action_scale": 1.0},
        {
            "name": "sensor_burst",
            "obs_scale": mid,
            "obs_noise": 0.03 * hi,
            "action_scale": 1.0,
        },
        {
            "name": "actuator_misgain",
            "obs_scale": 1.0,
            "action_scale": min(mid, 8.0),
            "action_noise": 0.02 * hi,
        },
        {
            "name": "sensor_actuator_chaos",
            "obs_scale": mid,
            "obs_noise": 0.02 * hi,
            "action_scale": min(mid, 8.0),
            "action_noise": 0.02 * hi,
        },
    ]

    if "Pendulum" in env_name:
        physical = [
            {"name": "mass_extreme", "mass": hi, "obs_scale": 1.0, "action_scale": 1.0},
            {"name": "gravity_extreme", "gravity": hi, "obs_scale": 1.0, "action_scale": 1.0},
            {"name": "length_extreme", "length": mid, "obs_scale": 1.0, "action_scale": 1.0},
            {
                "name": "compound_physics_chaos",
                "mass": mid,
                "gravity": mid,
                "length": mid,
                "obs_noise": 0.01 * hi,
                "action_scale": min(mid, 8.0),
            },
        ]
    elif any(token in env_name for token in ("Hopper", "Walker", "HalfCheetah", "Ant", "Humanoid")):
        physical = [
            {"name": "body_mass_extreme", "body_mass": hi, "obs_scale": 1.0, "action_scale": 1.0},
            {"name": "friction_collapse", "friction": lo, "obs_scale": 1.0, "action_scale": 1.0},
            {"name": "gravity_extreme", "gravity": min(hi, 20.0), "obs_scale": 1.0, "action_scale": 1.0},
            {
                "name": "compound_physics_chaos",
                "body_mass": mid,
                "friction": max(lo, 0.02),
                "gravity": min(mid, 10.0),
                "obs_noise": 0.01 * hi,
                "action_scale": min(mid, 8.0),
            },
        ]
    else:
        physical = []

    return common[:1] + physical + common[1:]


def default_structured_profiles(env_name: Optional[str] = None, intensity: float = 50.0) -> List[Profile]:
    """
    Build structured bus-like OOD regimes for generic RL.

    The original ``extreme`` family deliberately mixes physics, sensor, and
    actuator shocks. That is a useful stress test, but it is not analogous to
    the bus benchmark where a low-dimensional latent demand regime coherently
    changes the dynamics. This profile family keeps the OOD magnitude large
    while making each profile a coherent latent-load mode. No oracle label is
    exposed to the agent; the name is only stored in info/logging.
    """
    hi = max(float(intensity), 1.0)
    mid = float(np.sqrt(hi))
    lo = 1.0 / mid
    env_name = env_name or ""

    def base(name, latent_load, **kwargs):
        out = {
            "name": name,
            "latent_load": float(latent_load),
            "obs_scale": 1.0,
            "action_scale": 1.0,
        }
        out.update(kwargs)
        return out

    if "Pendulum" in env_name:
        def p(name, load):
            # High demand means heavier/slower pendulum and stronger gravity.
            # Exponents keep intensity=50 severe but not numerically absurd.
            return base(
                name,
                load,
                mass=float(load ** 0.25),
                gravity=float(load ** 0.50),
                length=float(load ** 0.25),
            )
        profiles = [
            base("id", 1.0),
            p("low_load", lo),
            p("rush_load", mid),
            p("mega_event", hi),
            base(
                "recovery_aftershock",
                mid,
                mass=float(mid ** 0.25),
                gravity=float(hi ** 0.25),
                length=float(lo ** 0.25),
            ),
        ]
    elif any(token in env_name for token in ("Hopper", "Walker", "HalfCheetah", "Ant", "Humanoid")):
        def m(name, load):
            # Coherent terrain/body-load mode: mass and gravity rise together,
            # friction drops for high-load regimes. This preserves structure
            # while remaining far outside ordinary domain randomization.
            return base(
                name,
                load,
                body_mass=float(load ** 0.33),
                gravity=float(min(load ** 0.25, 3.0)),
                friction=float(max(load ** -0.25, 0.25)),
            )
        profiles = [
            base("id", 1.0),
            m("easy_surface", lo),
            m("heavy_traffic", mid),
            m("mega_event", hi),
            base(
                "slippery_rush",
                mid,
                body_mass=float(mid ** 0.25),
                gravity=float(min(mid ** 0.20, 2.0)),
                friction=float(max(hi ** -0.33, 0.15)),
            ),
        ]
    else:
        profiles = [
            base("id", 1.0),
            base("low_load", lo, obs_scale=float(lo ** 0.25)),
            base("rush_load", mid, obs_scale=float(mid ** 0.25)),
            base("mega_event", hi, obs_scale=float(hi ** 0.25)),
        ]

    return profiles


def default_load_profiles(env_name: Optional[str] = None, intensity: float = 50.0) -> List[Profile]:
    """
    Build bus-like latent-load regimes that are severe but coherent.

    The ``structured`` family mostly scales global physics. That can make
    robust locomotion enough, leaving little value in identifying the current
    regime. This family treats OOD as a hidden load variable: it jointly
    changes mass/terrain/actuation and, for locomotion tasks, adds a
    profile-dependent external force and velocity preference. The profile name
    is only exposed in ``info`` for diagnostics.
    """
    hi = max(float(intensity), 1.0)
    if hi <= 1.0:
        stress = 0.0
    else:
        stress = np.log(hi) / np.log(50.0)
    stress = float(np.clip(stress, 0.0, 1.5))
    env_name = env_name or ""

    def base(name, latent_load, **kwargs):
        out = {
            "name": name,
            "latent_load": float(latent_load),
            "obs_scale": 1.0,
            "action_scale": 1.0,
        }
        out.update(kwargs)
        return out

    if "Pendulum" in env_name:
        def p(name, z):
            return base(
                name,
                z,
                mass=float(max(0.55, 1.0 + 0.80 * z)),
                gravity=float(max(0.55, 1.0 + 1.00 * z)),
                length=float(max(0.75, 1.0 + 0.12 * z)),
                action_scale=float(np.clip(1.0 - 0.10 * z, 0.75, 1.12)),
                obs_noise=float(max(z, 0.0) * 0.004),
                extra_control_cost=float(max(z, 0.0) * 0.0004),
            )

        profiles = [
            base("id", 0.0),
            p("low_load", -0.25 * stress),
            p("rush_load", 0.45 * stress),
            p("mega_event", 0.85 * stress),
            base(
                "recovery_aftershock",
                0.35 * stress,
                mass=float(1.0 + 0.35 * stress),
                gravity=float(1.0 + 0.60 * stress),
                length=float(max(0.85, 1.0 - 0.10 * stress)),
                action_scale=float(np.clip(1.0 - 0.05 * stress, 0.80, 1.0)),
                obs_noise=float(0.003 * stress),
                extra_control_cost=float(0.0003 * stress),
            ),
        ]
    elif any(token in env_name for token in ("Hopper", "Walker", "HalfCheetah", "Ant", "Humanoid")):
        def m(name, z):
            return base(
                name,
                z,
                body_mass=float(max(0.55, 1.0 + 0.82 * z)),
                gravity=float(max(0.75, 1.0 + 0.18 * z)),
                friction=float(np.clip(1.0 - 0.35 * z, 0.35, 1.35)),
                action_scale=float(np.clip(1.0 - 0.18 * z, 0.60, 1.15)),
                x_force=float(-22.0 * z),
                target_x_velocity=float(np.clip(1.15 - 0.34 * z, 0.35, 1.50)),
                target_velocity_weight=float(0.18 * abs(z)),
                extra_control_cost=float(0.0008 * max(z, 0.0)),
            )

        profiles = [
            base("id", 0.0),
            m("easy_surface", -0.35 * stress),
            m("rush_load", 0.75 * stress),
            m("mega_event", 1.30 * stress),
            base(
                "slippery_rush",
                0.95 * stress,
                body_mass=float(1.0 + 0.45 * stress),
                gravity=float(1.0 + 0.12 * stress),
                friction=float(np.clip(1.0 - 0.65 * stress, 0.25, 1.0)),
                action_scale=float(np.clip(1.0 - 0.12 * stress, 0.65, 1.0)),
                x_force=float(-16.0 * stress),
                target_x_velocity=float(np.clip(0.95 - 0.22 * stress, 0.40, 1.0)),
                target_velocity_weight=float(0.15 * stress),
                extra_control_cost=float(0.0006 * stress),
            ),
        ]
    else:
        profiles = [
            base("id", 0.0),
            base("low_load", -0.35 * stress, obs_scale=float(1.0 - 0.10 * stress)),
            base("rush_load", 0.75 * stress, obs_scale=float(1.0 + 0.15 * stress)),
            base("mega_event", 1.30 * stress, obs_scale=float(1.0 + 0.30 * stress)),
        ]

    return profiles


def make_ood_profiles(
    env_name: Optional[str] = None,
    intensity: float = 50.0,
    family: str = "extreme",
) -> List[Profile]:
    family = str(family or "extreme")
    if family == "extreme":
        return default_extreme_profiles(env_name, intensity=intensity)
    if family == "structured":
        return default_structured_profiles(env_name, intensity=intensity)
    if family in ("load", "latent_load"):
        return default_load_profiles(env_name, intensity=intensity)
    raise ValueError(f"Unknown OOD profile family: {family}")


class ExtremeOODWrapper(gym.Wrapper):
    """
    Apply severe OOD perturbations under a configurable schedule.

    Supported schedule values:
      - ``static``: hold one profile for the whole episode.
      - ``episode_random``: sample one non-ID profile per episode.
      - ``abrupt``: start ID, switch once at ``abrupt_step``.
      - ``oscillating``: cycle through profiles every ``switch_every_steps``.
      - ``step_random``: resample every ``switch_every_steps``.
      - ``markov``: dwell in a profile and randomly switch at period ticks.
    """

    def __init__(
        self,
        env,
        profiles: Optional[Iterable[Profile]] = None,
        config: Optional[ExtremeOODConfig] = None,
        **config_kwargs,
    ):
        super().__init__(env)
        if config is None:
            config = ExtremeOODConfig(**config_kwargs)
        elif config_kwargs:
            raise ValueError("Pass either config or config keyword arguments, not both")
        self.config = config

        env_name = getattr(getattr(env, "spec", None), "id", None)
        self.profiles = list(profiles) if profiles is not None else make_ood_profiles(
            env_name, intensity=config.intensity, family=config.profile_family
        )
        if not self.profiles:
            self.profiles = [{"name": "id"}]
        self._profile_by_name = {str(p.get("name", i)): p for i, p in enumerate(self.profiles)}
        self._rng = np.random.RandomState(config.seed)
        self._step = 0
        self._episode = -1
        self._base_captured = False
        self._base: Dict[str, Any] = {}
        self.current_profile: Profile = self.profiles[0]

    @property
    def active_profile_name(self) -> str:
        return str(self.current_profile.get("name", "unnamed"))

    def set_profile(self, profile_name_or_dict):
        """Force a profile immediately; useful for deterministic sweeps."""
        profile = self._coerce_profile(profile_name_or_dict)
        self._activate_profile(profile)

    def reset(self, **kwargs):
        self._step = 0
        self._episode += 1
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
        else:  # old gym
            obs, info = out, {}

        self._capture_base_params()
        self._activate_profile(self._initial_profile_for_episode())
        obs = self._transform_obs(obs)
        info = dict(info)
        info["ood_profile"] = self.active_profile_name
        info["ood_schedule"] = self.config.schedule
        info["ood_profile_family"] = self.config.profile_family
        info["ood_latent_load"] = self.current_profile.get("latent_load")
        return obs, info

    def step(self, action):
        self._maybe_switch_before_step()
        action = self._transform_action(action)
        self._apply_runtime_disturbance()
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:  # old gym
            obs, reward, done, info = out
            terminated, truncated = done, False

        reward = self._transform_reward(reward, action=action, info=info)
        obs = self._transform_obs(obs)
        info = dict(info)
        info["ood_profile"] = self.active_profile_name
        info["ood_schedule"] = self.config.schedule
        info["ood_step"] = self._step
        info["ood_profile_family"] = self.config.profile_family
        info["ood_latent_load"] = self.current_profile.get("latent_load")
        self._step += 1
        return obs, reward, terminated, truncated, info

    def _coerce_profile(self, profile_name_or_dict) -> Profile:
        if isinstance(profile_name_or_dict, dict):
            return dict(profile_name_or_dict)
        key = str(profile_name_or_dict)
        if key not in self._profile_by_name:
            raise KeyError(f"Unknown OOD profile '{key}'. Available: {list(self._profile_by_name)}")
        return self._profile_by_name[key]

    def _non_id_profiles(self) -> List[Profile]:
        non_id = [p for p in self.profiles if str(p.get("name", "")) != "id"]
        return non_id or self.profiles

    def _sample_profile(self) -> Profile:
        profiles = self._non_id_profiles()
        return profiles[int(self._rng.randint(0, len(profiles)))]

    def _initial_profile_for_episode(self) -> Profile:
        schedule = self.config.schedule
        if schedule == "static":
            if self.config.static_profile is not None:
                return self._coerce_profile(self.config.static_profile)
            return self.profiles[min(1, len(self.profiles) - 1)]
        if schedule == "episode_random":
            return self._sample_profile()
        if schedule in ("abrupt", "oscillating", "step_random", "markov"):
            return self._profile_by_name.get("id", self.profiles[0])
        raise ValueError(f"Unknown OOD schedule: {schedule}")

    def _maybe_switch_before_step(self):
        schedule = self.config.schedule
        if schedule == "abrupt" and self._step == self.config.abrupt_step:
            self._activate_profile(self._sample_profile())
        elif schedule == "oscillating":
            period = max(int(self.config.switch_every_steps), 1)
            index = (self._step // period) % len(self.profiles)
            self._activate_profile(self.profiles[index])
        elif schedule == "step_random":
            period = max(int(self.config.switch_every_steps), 1)
            if self._step > 0 and self._step % period == 0:
                self._activate_profile(self._sample_profile())
        elif schedule == "markov":
            period = max(int(self.config.switch_every_steps), 1)
            switch_p = float(np.clip(self.config.switch_probability, 0.0, 1.0))
            if self._step > 0 and self._step % period == 0 and self._rng.rand() < switch_p:
                candidates = [
                    p for p in self.profiles
                    if str(p.get("name", "")) != self.active_profile_name
                ]
                if candidates:
                    idx = int(self._rng.randint(0, len(candidates)))
                    self._activate_profile(candidates[idx])

    def _capture_base_params(self):
        if self._base_captured:
            return
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "model"):
            model = unwrapped.model
            self._base["body_mass"] = model.body_mass.copy()
            self._base["geom_friction"] = model.geom_friction.copy()
            self._base["gravity"] = model.opt.gravity.copy()
            self._base["force_body_id"] = 1 if getattr(model, "nbody", 0) > 1 else 0
        if hasattr(unwrapped, "m") and hasattr(unwrapped, "g") and hasattr(unwrapped, "l"):
            self._base["pendulum_m"] = float(unwrapped.m)
            self._base["pendulum_g"] = float(unwrapped.g)
            self._base["pendulum_l"] = float(unwrapped.l)
        self._base_captured = True

    def _activate_profile(self, profile: Profile):
        self.current_profile = dict(profile)
        self._apply_physics_profile(self.current_profile)

    def _apply_physics_profile(self, profile: Profile):
        if not self._base_captured:
            return
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "model") and "body_mass" in self._base:
            model = unwrapped.model
            body_mult = _safe_float(profile.get("body_mass", profile.get("mass", 1.0)))
            friction_mult = _safe_float(profile.get("friction", 1.0))
            gravity_mult = _safe_float(profile.get("gravity", 1.0))
            model.body_mass[:] = self._base["body_mass"] * body_mult
            model.geom_friction[:] = self._base["geom_friction"] * friction_mult
            model.opt.gravity[:] = self._base["gravity"] * gravity_mult
        if "pendulum_m" in self._base:
            unwrapped.m = self._base["pendulum_m"] * _safe_float(profile.get("mass", 1.0))
            unwrapped.g = self._base["pendulum_g"] * _safe_float(profile.get("gravity", 1.0))
            unwrapped.l = self._base["pendulum_l"] * _safe_float(profile.get("length", 1.0))

    def _transform_obs(self, obs):
        obs_arr = np.asarray(obs, dtype=np.float32)
        scale = _safe_float(self.current_profile.get("obs_scale", 1.0))
        obs_arr = obs_arr * scale
        noise = _safe_float(self.current_profile.get("obs_noise", 0.0), default=0.0)
        if self.config.include_sensor_noise and noise > 0:
            obs_arr = obs_arr + self._rng.normal(0.0, noise, size=obs_arr.shape).astype(np.float32)
        return _finite_box_clip(self.observation_space, obs_arr)

    def _transform_action(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        scale = _safe_float(self.current_profile.get("action_scale", 1.0))
        action_arr = action_arr * scale
        noise = _safe_float(self.current_profile.get("action_noise", 0.0), default=0.0)
        if self.config.include_action_noise and noise > 0:
            width = 1.0
            if hasattr(self.action_space, "high") and hasattr(self.action_space, "low"):
                high = np.asarray(self.action_space.high)
                low = np.asarray(self.action_space.low)
                if np.all(np.isfinite(high)) and np.all(np.isfinite(low)):
                    width = np.maximum(high - low, 1e-6)
            action_arr = action_arr + self._rng.normal(0.0, noise, size=action_arr.shape) * width
        return _finite_box_clip(self.action_space, action_arr)

    def _apply_runtime_disturbance(self):
        unwrapped = self.env.unwrapped
        data = getattr(unwrapped, "data", None)
        if data is None or not hasattr(data, "xfrc_applied"):
            return
        xfrc = data.xfrc_applied
        if xfrc is None or xfrc.ndim != 2 or xfrc.shape[1] < 6:
            return
        xfrc[:] = 0.0
        fx = _safe_float(self.current_profile.get("x_force", 0.0), default=0.0)
        fz = _safe_float(self.current_profile.get("z_force", 0.0), default=0.0)
        if fx == 0.0 and fz == 0.0:
            return
        default_body = int(self._base.get("force_body_id", min(1, xfrc.shape[0] - 1)))
        body_id = int(self.current_profile.get("force_body_id", default_body))
        body_id = int(np.clip(body_id, 0, xfrc.shape[0] - 1))
        xfrc[body_id, 0] = fx
        xfrc[body_id, 2] = fz

    def _transform_reward(self, reward, action=None, info=None):
        reward_value = float(reward)
        reward_value *= _safe_float(self.current_profile.get("reward_scale", 1.0))
        target_velocity = self.current_profile.get("target_x_velocity")
        if target_velocity is not None and info is not None:
            velocity = info.get("x_velocity", info.get("forward_velocity"))
            if velocity is not None:
                weight = _safe_float(
                    self.current_profile.get("target_velocity_weight", 0.0),
                    default=0.0,
                )
                reward_value -= weight * float(float(velocity) - float(target_velocity)) ** 2
        control_cost = _safe_float(self.current_profile.get("extra_control_cost", 0.0), default=0.0)
        if control_cost > 0 and action is not None:
            action_arr = np.asarray(action, dtype=np.float32)
            reward_value -= control_cost * float(np.sum(np.square(action_arr)))
        noise = _safe_float(self.current_profile.get("reward_noise", 0.0), default=0.0)
        if self.config.include_reward_noise and noise > 0:
            reward_value += float(self._rng.normal(0.0, noise))
        return reward_value


def make_extreme_ood_env(env_name: str, **kwargs):
    """Construct ``gym.make(env_name)`` wrapped with ``ExtremeOODWrapper``."""
    env = gym.make(env_name)
    return ExtremeOODWrapper(env, **kwargs)
