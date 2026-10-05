# Nonlinear load v26: frozen critic-direction diagnostic

Reuse v25 uniform/early last models for seeds 19/20 from t129046/t129047. No training, fitting, new seeds or manuscript changes. Each seed's models run together on one scheduler-selected compatible Linux GPU; no placement preferences, pins or Windows nodes.

Keep v24's protocol: radii 0.2/1, loads 1/50, paths 60000--60099, Gaussian-noise seeds 92000--92007, gamma 0.99, 200 steps and first-step latent-mean perturbations +/-0.01 and +/-0.005 toward each model's own fitted-reference mean. The initial standard deviation is unchanged; later actions follow that model's frozen stochastic policy with shared Gaussian draws and disturbance paths.

Export only actor/critic parameters, fitted gains, terminal alpha/reward-EMA standard deviation/belief coefficient and selected source cost vectors. First reproduce native deterministic actor costs for the same paths (rtol 1e-4, atol 1e-3). Check source seed/arm/protocol, completed 500-episode budget and terminal scales against v25 evaluation records before evaluating directions.

Compare Q-mean derivatives with normalized discounted reward and soft-Q return; compare unpenalized and risk-adjusted actor scores separately with soft actor return. Entropy starts at step 1 for soft-Q return and includes step 0 for the actor objective; terminate without a bootstrap at step 199. Retain per-path signs, analytic/finite-difference comparisons and per-noise-draw means at both step sizes. Risk is a separate prediction penalty, not an observed risk-adjusted return target.

Formal budget: 64,000 stochastic controller episodes plus 1,600 deterministic reproduction episodes across two paired tasks. Seed 19 preflight: 40 stochastic plus 4 reproduction episodes. Optimizer updates and identification interactions are zero. Full draw/path matrices remain in remote raw_outputs; only compact result JSON is synchronized. No checkpoint or replay pull.

## Decision And Limitations

Use this diagnostic to decide one next critic intervention, not a seed/baseline expansion or a retrospective new pass gate. Historical HOLD remains unchanged. Each model has its own actor, reward scale and unit direction; cross-model derivative magnitudes are not same-action causal effects. Eight draws and reused paths support local ordering analysis, not proof of global calibration or general RL superiority. Reduce ordering error only when finite-difference signs are stable; do not select an intervention from a single seed or aggregate cost alone.
