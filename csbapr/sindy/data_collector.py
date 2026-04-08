"""
Trajectory Collection and State Derivative Computation for SINDy.

⚠️ SINDy identifies ENVIRONMENT DYNAMICS, not the policy.
   X_dot = (s_{t+1} - s_t) / dt is the finite-difference derivative estimate.
   
⚠️ dt must match the environment's actual timestep (trap #9).
   MuJoCo: dt=env.dt (typically 0.002-0.05)
   Gymnasium: dt=env.dt or 1.0 (step-level)
"""

import numpy as np


def collect_trajectories(env, policy=None, n_episodes: int = 50,
                          max_steps: int = 500, dt: float = 1.0,
                          seed: int = None):
    """
    Collect trajectories from an environment for SINDy identification.
    
    Args:
        env: Gymnasium environment
        policy: Optional policy function s → a. If None, uses random actions.
        n_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        dt: Environment time step (trap #9: MUST match env.dt)
        seed: Random seed
    
    Returns:
        X_list: List of state arrays [T, n_state] (one per trajectory)
        U_list: List of action arrays [T, n_action] (one per trajectory)
        X_dot_list: List of state derivative arrays [T-1, n_state] (for continuous SINDy)
    """
    X_list = []
    U_list = []
    X_dot_list = []

    for ep in range(n_episodes):
        states = []
        actions = []
        derivatives = []

        state, info = env.reset(seed=seed + ep if seed else None)
        states.append(state)

        for step in range(max_steps):
            if policy is not None:
                action = policy(state)
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Ensure action is 1D array (flatten scalar or multi-dim)
            action_flat = np.atleast_1d(np.asarray(action, dtype=np.float64))
            actions.append(action_flat)
            derivatives.append((next_state - state) / dt)
            states.append(next_state)

            state = next_state
            if done:
                break

        X_list.append(np.array(states))
        U_list.append(np.array(actions))
        if len(derivatives) > 0:
            X_dot_list.append(np.array(derivatives))

    return X_list, U_list, X_dot_list


def compute_state_derivatives(X_list, dt: float = 1.0):
    """
    Compute state derivatives from trajectory data via finite differences.
    
    Args:
        X_list: List of state trajectories [T, n_state]
        dt: Time step
    
    Returns:
        X_flat: Stacked states [N, n_state]
        X_dot_flat: Stacked derivatives [N, n_state]
    """
    X_flat = []
    X_dot_flat = []

    for traj in X_list:
        for t in range(len(traj) - 1):
            X_flat.append(traj[t])
            X_dot_flat.append((traj[t + 1] - traj[t]) / dt)

    return np.array(X_flat), np.array(X_dot_flat)


def prepare_sindy_data_discrete(X_list, U_list):
    """
    Prepare data for discrete-time SINDy (x_{t+1} = f(x_t, u_t)).
    
    PySINDy's discrete_time=True expects state trajectories directly.
    No derivative computation needed.
    
    Args:
        X_list: List of state arrays [T, n_state]
        U_list: List of action arrays [T-1, n_action]
    
    Returns:
        X_list, U_list (trimmed to match lengths)
    """
    X_trimmed = []
    U_trimmed = []

    for X, U in zip(X_list, U_list):
        # Ensure X has one more timestep than U
        min_len = min(len(X) - 1, len(U))
        X_trimmed.append(X[:min_len + 1])
        U_trimmed.append(U[:min_len])

    return X_trimmed, U_trimmed
