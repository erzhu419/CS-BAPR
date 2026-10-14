"""Feedback from fitted physical dynamics; force innovations stay stochastic."""

import numpy as np
from scipy.linalg import solve_discrete_are

from csbapr.envs.lqr_env import Q, R, ACTION_SCALE, get_optimal_gain
from csbapr.sindy.feedback import LQRFeedbackReference


def known_load_reference():
    return LQRFeedbackReference(np.hstack((get_optimal_gain(), np.eye(2) / ACTION_SCALE)))


def fitted_load_reference(model):
    if (model.n_state, model.n_control, model.poly_degree, model.discrete_time) != (6, 2, 1, True):
        raise ValueError("Random-load feedback requires the six-state linear controlled increment fit")
    powers = model.library.powers_
    physical_coefficients = model.coeffs[:4]
    if np.max(np.abs(physical_coefficients[:, powers.sum(axis=1) == 0])) > 1e-6:
        raise ValueError("Physical dynamics must have zero intercept")
    linear = physical_coefficients[:, powers.sum(axis=1) == 1]
    A = np.eye(4) + linear[:, :4]
    D, B = linear[:, 4:6], linear[:, 6:]
    action_cost = R * ACTION_SCALE ** 2
    P = solve_discrete_are(A, B, Q, action_cost)
    K = np.linalg.solve(action_cost + B.T @ P @ B, B.T @ P @ A)
    cancellation = np.linalg.lstsq(B, D, rcond=None)[0]
    # Reuse the existing gain-only reference and its checkpoint serialization.
    return LQRFeedbackReference(np.hstack((K, cancellation)))
