"""Always-active bounded heavy-tail load, with unchanged plant and cost."""

import numpy as np

from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv

BASE_FORCE_CAP = 0.08


class SustainedLoadEnv(NonlinearLoadEnv):
    def __init__(self, severity=1.0, seed=None):
        super().__init__(severity, seed)
        self.force_cap = BASE_FORCE_CAP * self.severity
        self.storm_probability = 1.0

    @staticmethod
    def specification():
        settings = NonlinearLoadEnv.specification()
        settings.update({
            "environment": "SustainedNonlinearLoad-v0",
            "base_force_cap": BASE_FORCE_CAP,
            "base_storm_probability": 1.0,
            "storm_probability_rule": "always-active load",
            "tail_rule": "0.5 + 0.5 * min(1, 0.1 * (1 + Pareto(tail_shape)))",
        })
        del settings["background_rule"]
        return settings

    def _draw_disturbance(self):
        # Keep latent draws paired with the old process; only their load mapping changes.
        self._rng.random()
        self._rng.uniform(-0.1, 0.1, 2)
        signs = self._rng.choice([-1.0, 1.0], 2)
        tail = np.minimum(1.0, 0.1 * (1.0 + self._rng.pareto(1.5, 2)))
        self.disturbance = self.force_cap * signs * (0.5 + 0.5 * tail)
        self.is_storm = True
