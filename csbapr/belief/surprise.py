"""
SurpriseComputer — Multi-signal surprise detection for CS-BAPR.

Directly adapted from BAPR/bapr_components.py (proven convergent).
Corresponds to CSBAPR.lean §8: κ-driven Bayesian update.
"""

import torch
import numpy as np
from collections import deque


class SurpriseComputer:
    """
    Multi-signal surprise detector.
    Combines reward z-score and Q-value std change to detect environment shifts.
    """
    def __init__(self, ema_alpha: float = 0.3, reward_window: int = 50):
        self.ema_alpha = ema_alpha
        self.ema_surprise = 0.0
        self.reward_window = reward_window
        self.reward_history = deque(maxlen=reward_window)
        self.reward_ema = 0.0
        self.reward_var_ema = 1.0
        self.prev_q_std = None

    def reset(self):
        self.ema_surprise = 0.0
        self.reward_history = deque(maxlen=self.reward_window)
        self.reward_ema = 0.0
        self.reward_var_ema = 1.0
        self.prev_q_std = None

    def compute(self, reward_batch: torch.Tensor,
                q_std: torch.Tensor,
                reg_norm_current: torch.Tensor = None,
                reg_norm_target: torch.Tensor = None) -> float:
        """
        Compute surprise signal from multiple sources.
        
        Args:
            reward_batch: [batch, 1] rewards
            q_std: [batch] or scalar Q-value ensemble std
            reg_norm_current: current critic reg norm (optional)
            reg_norm_target: target critic reg norm (optional)
        
        Returns:
            EMA-smoothed surprise signal (scalar)
        """
        signals = []

        # Signal 1: Reward z-score
        batch_reward_mean = reward_batch.mean().item()
        self.reward_history.append(batch_reward_mean)  # deque auto-caps at reward_window

        self.reward_ema = 0.9 * self.reward_ema + 0.1 * batch_reward_mean
        deviation = (batch_reward_mean - self.reward_ema) ** 2
        self.reward_var_ema = 0.9 * self.reward_var_ema + 0.1 * deviation
        reward_std = max(self.reward_var_ema ** 0.5, 1e-6)
        reward_zscore = abs(batch_reward_mean - self.reward_ema) / reward_std
        signals.append(reward_zscore)

        # Signal 2: Q-std spike (one-sided; per BAPR v14 / GPT-5.5 review v2)
        # Old version used abs(...) which also fired when ensemble was
        # *converging* (Q-std dropping), polluting BOCD with false positives.
        # The actual signal of regime change is Q-std *spiking up*, so we keep
        # only the positive deviation.
        current_q_std = q_std.mean().item() if isinstance(q_std, torch.Tensor) else q_std
        if self.prev_q_std is not None and self.prev_q_std > 1e-6:
            q_std_spike = max(current_q_std / self.prev_q_std - 1.0, 0.0)
            signals.append(q_std_spike)
        self.prev_q_std = current_q_std

        # Signal 3: reg_norm divergence
        if reg_norm_current is not None and reg_norm_target is not None:
            reg_raw = (reg_norm_current - reg_norm_target).abs().mean().item()
            signals.append(reg_raw * 0.5)

        raw_surprise = max(signals) if signals else 0.0
        self.ema_surprise = self.ema_alpha * raw_surprise + (1 - self.ema_alpha) * self.ema_surprise
        return self.ema_surprise
