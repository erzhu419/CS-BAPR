"""Reward, critic-disagreement and critic-drift surprise observations."""

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
        self.last_components = {}

    def reset(self):
        self.ema_surprise = 0.0
        self.reward_history = deque(maxlen=self.reward_window)
        self.reward_ema = 0.0
        self.reward_var_ema = 1.0
        self.prev_q_std = None
        self.last_components = {}

    def _reg_signal(self, absolute, relative):
        return 0.5 * absolute

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
        # Signal 1: Reward z-score
        batch_reward_mean = reward_batch.mean().item()
        self.reward_history.append(batch_reward_mean)  # deque auto-caps at reward_window

        self.reward_ema = 0.9 * self.reward_ema + 0.1 * batch_reward_mean
        deviation = (batch_reward_mean - self.reward_ema) ** 2
        self.reward_var_ema = 0.9 * self.reward_var_ema + 0.1 * deviation
        reward_std = max(self.reward_var_ema ** 0.5, 1e-6)
        reward_zscore = abs(batch_reward_mean - self.reward_ema) / reward_std

        # Signal 2: Q-std spike (one-sided; per BAPR v14 / GPT-5.5 review v2)
        # Old version used abs(...) which also fired when ensemble was
        # *converging* (Q-std dropping), polluting BOCD with false positives.
        # The actual signal of regime change is Q-std *spiking up*, so we keep
        # only the positive deviation.
        current_q_std = q_std.mean().item() if isinstance(q_std, torch.Tensor) else q_std
        q_std_spike = 0.0
        if self.prev_q_std is not None and self.prev_q_std > 1e-6:
            q_std_spike = max(current_q_std / self.prev_q_std - 1.0, 0.0)
        self.prev_q_std = current_q_std

        # Signal 3: reg_norm divergence
        reg_absolute, reg_relative, reg_current, reg_target = 0.0, 0.0, 0.0, 0.0
        if reg_norm_current is not None and reg_norm_target is not None:
            difference = (reg_norm_current - reg_norm_target).abs()
            reg_absolute = difference.mean().item()
            reg_relative = (difference / reg_norm_target.abs().clamp_min(1e-6)).mean().item()
            reg_current = reg_norm_current.mean().item()
            reg_target = reg_norm_target.mean().item()

        reg_signal = self._reg_signal(reg_absolute, reg_relative)
        signals = (reward_zscore, q_std_spike, reg_signal)
        raw_surprise = max(signals)
        dominant = int(np.argmax(signals))
        self.last_components = {
            'surprise_reward_zscore': reward_zscore,
            'surprise_q_std_spike': q_std_spike,
            'surprise_reg_absolute': reg_absolute,
            'surprise_reg_relative': reg_relative,
            'surprise_reg_selected': reg_signal,
            'surprise_raw': raw_surprise,
            'surprise_reward_dominant': float(dominant == 0),
            'surprise_q_std_dominant': float(dominant == 1),
            'surprise_reg_dominant': float(dominant == 2),
            'signal_q_std': float(current_q_std),
            'critic_reg_norm_current': reg_current,
            'critic_reg_norm_target': reg_target,
        }
        self.ema_surprise = self.ema_alpha * raw_surprise + (1 - self.ema_alpha) * self.ema_surprise
        return self.ema_surprise


class RelativeSurpriseComputer(SurpriseComputer):
    """Use per-critic relative L1 drift; all other signals and EMA stay fixed."""

    def _reg_signal(self, absolute, relative):
        return 0.5 * relative
