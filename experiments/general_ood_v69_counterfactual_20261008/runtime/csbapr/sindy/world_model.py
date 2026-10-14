"""Sparse state-increment or continuous-time dynamics with explicit controls."""

import numpy as np
import pysindy as ps


class SymbolicWorldModel:
    def __init__(self, n_state: int, n_control: int = 0,
                 poly_degree: int = 2, threshold: float = 0.1,
                 discrete_time: bool = True, feature_names: list = None):
        self.n_state = n_state
        self.n_control = n_control
        self.discrete_time = discrete_time
        self.poly_degree = poly_degree
        self.threshold = threshold
        self.feature_names = feature_names
        self.coeffs = None
        self.sparsity = None
        self.library = ps.PolynomialLibrary(degree=poly_degree)
        self.optimizer = ps.STLSQ(threshold=threshold)
        self.model = ps.SINDy(feature_library=self.library, optimizer=self.optimizer)

    def training_arrays(self, X, U=None, X_dot=None, multiple_trajectories=False):
        """Align each transition before concatenation; never difference across resets."""
        xs = X if multiple_trajectories else [X]
        us = U if multiple_trajectories else [U]
        ys = X_dot if multiple_trajectories else [X_dot]
        if U is None:
            us = [None] * len(xs)
        if X_dot is None:
            ys = [None] * len(xs)
        if not (len(xs) == len(us) == len(ys)):
            raise ValueError("State, control and target trajectory counts must match")
        states, controls, targets = [], [], []
        for x, u, y in zip(xs, us, ys):
            x = np.asarray(x, dtype=float)
            if y is None:
                if not self.discrete_time:
                    raise ValueError("Continuous-time fitting requires explicit derivatives")
                y = np.diff(x, axis=0)
                x = x[:-1]
            else:
                y = np.asarray(y, dtype=float)
                if len(x) == len(y) + 1:
                    x = x[:-1]
            if x.ndim != 2 or x.shape != y.shape or x.shape[1] != self.n_state:
                raise ValueError("States and dynamics targets must be aligned [N, n_state]")
            states.append(x)
            targets.append(y)
            if self.n_control:
                if u is None:
                    raise ValueError("Control-aware fitting requires actions")
                u = np.asarray(u, dtype=float).reshape(-1, self.n_control)
                if len(u) != len(x):
                    raise ValueError("Each transition requires its own aligned action")
                controls.append(u)
        return (np.vstack(states),
                np.vstack(controls) if self.n_control else None,
                np.vstack(targets))

    def fit(self, X, U=None, X_dot=None, t=None, multiple_trajectories=False):
        """Discrete targets are x_next - x, not x_next; u is in environment units."""
        x, u, y = self.training_arrays(X, U, X_dot, multiple_trajectories)
        self.model.fit(x, t=1.0 if t is None else t, x_dot=y, u=u,
                       feature_names=self.feature_names)
        self.coeffs = self.model.coefficients().copy()
        self.sparsity = float((np.abs(self.coeffs) > 1e-6).mean())
        return self

    def predict(self, x, u=None):
        if self.n_control and u is None:
            raise ValueError("Control-aware prediction requires actions")
        return self.model.predict(x, u=u if self.n_control else None)

    def prediction_error(self, states, targets, u=None):
        """Mean prediction error, not a derivative-error certificate."""
        return np.linalg.norm(self.predict(states, u) - targets, axis=-1).mean()

    def linear_dynamics(self):
        """Extract A, B only from a homogeneous degree-one discrete increment fit."""
        if not self.discrete_time or self.poly_degree != 1 or not self.n_control:
            raise ValueError("Feedback construction requires linear, discrete control dynamics")
        powers = self.library.powers_
        constant = (powers.sum(axis=1) == 0)
        if np.max(np.abs(self.coeffs[:, constant])) > 1e-6:
            raise ValueError("The LQR reference requires a zero-intercept dynamics fit")
        linear = self.coeffs[:, powers.sum(axis=1) == 1]
        return (np.eye(self.n_state) + linear[:, :self.n_state],
                linear[:, self.n_state:])

    def get_feature_names(self):
        return self.model.get_feature_names()

    def print_equations(self):
        self.model.print()
