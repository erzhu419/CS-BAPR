"""
Measure approximately-linear-in-state property of the trained CS-BAPR policy
on the bus benchmark (ID, m=1).

Method:
  1. Load v14 csbapr seed=7 (best 20x performer in the 10-seed sweep).
  2. Run 5 deterministic ID episodes, logging (state, hold) per decision.
  3. Restrict to interior decisions (0 < hold < 60) — the linear-form claim
     in the paper is for the interior regime, not the boundary clamps.
  4. Fit linear regression hold ~ A · state + b.
  5. Report R^2 on the held-out half.
"""
import os, sys, json
sys.path.insert(0, '/home/erzhu419/mine_code/CS-BAPR')
sys.path.insert(0, '/home/erzhu419/mine_code/offline-sumo/env')

import numpy as np
import torch

from sim_core.sim import MultiLineEnv, env_bus
from csbapr.config import CSBAPRConfig
from csbapr.agent import CSBAPRAgent

# Match the v14 csbapr config used during training
config = CSBAPRConfig()
config.state_dim = 15
config.action_dim = 1
config.hidden_dim = 64
config.num_critics = 3
config.batch_size = 256
config.use_nau_actor = True
config.actor_type = None
config.nau_reg_weight = 0.01
config.actor_weight_decay = 1e-4
config.beta_bc = 0.0
config.jac_weight = 0.0
config.weight_sym = 0.0

agent = CSBAPRAgent(config.state_dim, config.action_dim, config)
agent.load("/tmp/csbapr_seed7_best.pt")
print("Loaded csbapr seed7 best checkpoint")

env_bus._DATA_CACHE.clear()
env = MultiLineEnv("/home/erzhu419/mine_code/offline-sumo/env/calibrated_env", od_mult=1.0)
print(f"Loaded MultiLineEnv: {len(env.line_map)} lines")

# Reuse normalize_obs from test_multiline_convergence
sys.path.insert(0, '/home/erzhu419/mine_code/CS-BAPR/scripts')
from test_multiline_convergence import normalize_obs


def collect_one_episode(seed):
    obs_log = []
    hold_log = []
    env.reset()
    state_dict, _, _ = env.initialize_state()
    action_dict = {lid: {i: 0.0 for i in range(le.max_agent_num)}
                   for lid, le in env.line_map.items()}
    pending = {}
    for line_id, bus_dict in state_dict.items():
        for bus_id, lst in bus_dict.items():
            if not lst:
                continue
            sv = normalize_obs(lst[-1])
            raw = agent.select_action(sv, deterministic=True)
            hold = float((raw[0] + 1.0) / 2.0 * 60.0)
            hold = float(np.clip(hold, 0.0, 60.0))
            action_dict[line_id][bus_id] = hold
            pending[(line_id, bus_id)] = (sv, hold)
    done = False
    while not done:
        cur_state, _, done = env.step_to_event(action_dict)
        for lid in action_dict:
            for k in action_dict[lid]:
                action_dict[lid][k] = 0.0
        for line_id, bus_dict in cur_state.items():
            for bus_id, lst in bus_dict.items():
                if not lst:
                    continue
                sv_new = normalize_obs(lst[-1])
                key = (line_id, bus_id)
                if key in pending:
                    sv_old, hold_old = pending[key]
                    if int(sv_old[2] * 55) != int(sv_new[2] * 55):
                        # decision settled
                        obs_log.append(sv_old)
                        hold_log.append(hold_old)
                        pending.pop(key)
                raw = agent.select_action(sv_new, deterministic=True)
                hold = float((raw[0] + 1.0) / 2.0 * 60.0)
                hold = float(np.clip(hold, 0.0, 60.0))
                action_dict[line_id][bus_id] = hold
                pending[key] = (sv_new, hold)
    return np.array(obs_log), np.array(hold_log)


all_obs, all_hold = [], []
for ep_seed in [42, 43, 44, 45, 46]:
    obs, hold = collect_one_episode(ep_seed)
    all_obs.append(obs)
    all_hold.append(hold)
    print(f"  ep seed={ep_seed}: {len(obs)} decisions")
all_obs = np.concatenate(all_obs)
all_hold = np.concatenate(all_hold)
print(f"\nTotal decisions: {len(all_obs)}")

# Hold distribution
boundary_zero = (all_hold < 1e-6).sum()
boundary_max = (all_hold > 60 - 1e-3).sum()
interior = ((all_hold > 1e-6) & (all_hold < 60 - 1e-3)).sum()
print(f"Hold distribution: zero={boundary_zero} ({100*boundary_zero/len(all_hold):.1f}%), "
      f"interior={interior} ({100*interior/len(all_hold):.1f}%), "
      f"max={boundary_max} ({100*boundary_max/len(all_hold):.1f}%)")

# Fit linear regression on interior only
mask = (all_hold > 1e-6) & (all_hold < 60 - 1e-3)
X = all_obs[mask]
y = all_hold[mask]
print(f"\nFitting linear regression on {len(X)} interior decisions...")
n = len(X)
n_train = n // 2
rng = np.random.default_rng(0)
idx = rng.permutation(n)
X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

# Closed form: w = (X^T X)^-1 X^T y, with intercept
def fit_linreg(X, y):
    Xb = np.column_stack([X, np.ones(len(X))])
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return w

def r2(X, y, w):
    Xb = np.column_stack([X, np.ones(len(X))])
    yhat = Xb @ w
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

w = fit_linreg(X_train, y_train)
r2_train = r2(X_train, y_train, w)
r2_test = r2(X_test, y_test, w)
print(f"Linear fit: R² (train) = {r2_train:.3f}, R² (test) = {r2_test:.3f}")

# Also fit on ALL decisions (boundary included) for comparison
print(f"\n[For comparison] Linear fit on ALL {len(all_obs)} decisions (incl. boundary clamps):")
w_all = fit_linreg(all_obs[idx[:len(all_obs)//2]], all_hold[idx[:len(all_obs)//2]])
print(f"  R² (test) on full distribution = {r2(all_obs[idx[len(all_obs)//2:]], all_hold[idx[len(all_obs)//2:]], w_all):.3f}")
