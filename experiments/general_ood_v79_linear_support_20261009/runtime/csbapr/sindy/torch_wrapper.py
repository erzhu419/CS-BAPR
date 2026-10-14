"""Frozen polynomial dynamics with the fitted library's exact powers."""

import numpy as np
import torch
import torch.nn as nn


class SINDyTorchWrapper(nn.Module):
    def __init__(self, sindy_model):
        super().__init__()
        if sindy_model.coeffs is None:
            raise ValueError("SINDy model must be fitted before wrapping")
        self.n_state = sindy_model.n_state
        self.n_control = sindy_model.n_control
        self.poly_degree = sindy_model.poly_degree
        self.register_buffer("coeffs", torch.as_tensor(sindy_model.coeffs, dtype=torch.float32))
        self.register_buffer("powers", torch.as_tensor(sindy_model.library.powers_, dtype=torch.long))
        if self.powers.shape != (self.coeffs.shape[1], self.n_state + self.n_control):
            raise ValueError("Fitted polynomial powers and coefficients do not align")

    def forward(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        if self.n_control:
            if u is None:
                raise ValueError("Control-aware dynamics require actions")
            x = torch.cat([x, u], dim=-1)
        # Integer monomials preserve finite derivatives at zero, including degree zero.
        features = torch.ones(x.shape[0], self.powers.shape[0], dtype=x.dtype, device=x.device)
        for index in range(x.shape[1]):
            features = features * x[:, index:index + 1].pow(self.powers[:, index])
        return features @ self.coeffs.to(dtype=x.dtype).T

    @torch.no_grad()
    def validate_against_numpy(self, sindy_model, test_states: np.ndarray,
                               atol: float = 1e-4, test_controls: np.ndarray = None) -> bool:
        dtype, device = self.coeffs.dtype, self.coeffs.device
        x = torch.as_tensor(test_states, dtype=dtype, device=device)
        u = (torch.as_tensor(test_controls, dtype=dtype, device=device)
             if test_controls is not None else None)
        return bool(np.allclose(self(x, u).cpu().numpy(),
                                sindy_model.predict(test_states, test_controls),
                                atol=atol, rtol=0))
