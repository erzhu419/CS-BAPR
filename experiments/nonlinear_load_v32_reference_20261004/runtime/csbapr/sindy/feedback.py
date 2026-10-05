"""Action-valued, smooth feedback constructed from identified linear dynamics."""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are


class LQRFeedbackReference(nn.Module):
    """Tanh of the unconstrained LQR feedback, in normalized action units.

    The smoothed policy is a reference, not the constrained LQR optimum.
    Dynamics matrices come from the fit; cost matrices come from the task.
    """
    def __init__(self, gain):
        super().__init__()
        self.register_buffer("gain", torch.as_tensor(gain, dtype=torch.float32))

    @classmethod
    def from_world_model(cls, world_model, state_cost, action_cost):
        A, B = world_model.linear_dynamics()
        P = solve_discrete_are(A, B, state_cost, action_cost)
        K = np.linalg.solve(action_cost + B.T @ P @ B, B.T @ P @ A)
        return cls(K)

    def forward(self, states):
        return torch.tanh(-states @ self.gain.to(dtype=states.dtype).T)

    def compute_L_eff(self):
        norm = torch.linalg.matrix_norm(self.gain.double(), ord=2).item()
        return 4.0 / (3.0 * np.sqrt(3.0)) * norm ** 2
