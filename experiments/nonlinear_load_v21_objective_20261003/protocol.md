# Nonlinear load v21: frozen first-action objective diagnostic

Use v20 full-scope cubic-local seed15/16 last checkpoints (t126954/t126956), without training, fitting or seed expansion. v20 load-only is rejected; historical HOLD decisions and paper tables remain unchanged.

## Intervention

At each radius0.2/0.5/1 and severity1/50, recreate paired paths60000--60099. Compare the actor's deterministic first action with its own fitted-reference first action. Both branches then use the same frozen deterministic actor for steps1--199. Initial states and every applied disturbance must match across branches; no future disturbance information enters action selection. Each source yields1,200 controller episodes,2,400 total.

Report raw state/command costs, undiscounted and gamma0.99-discounted total costs, early50/late150 state costs, per-path changes and sign reversals. Reproduce the uninterrupted actor's saved v20 total/state/command costs at rtol1e-4/atol1e-3. These are first-action interventions, not full-reference policy comparisons or an optimality test.

At the same initial observations, record frozen ensemble Q mean/std, risk score Qmean+beta*Qstd with beta=-2-2*weighted_lambda from the terminal belief, and the pointwise soft surrogate risk_score-alpha*log_pi(action). Use the saved alpha and reward standard deviation; no belief or EMA update. Report critic action ranking and continuation ranking separately, per seed/radius/load, without pooling or a post hoc significance threshold.

## Verification And Dispatch

Tests must verify first-step-only replacement, gamma accumulation, exact exogenous-path coupling, source actor reproduction, native ensemble/std/entropy formulas, unchanged actor/critic parameters and preserved per-seed ordering signs. One CUDA preflight uses seed15, radius1/load50, paths60000/60001 (four episodes). Require source reproduction, finite values, cost decomposition and zero updates before dispatching two formal diagnostics. Scheduler selects idle compatible Linux GPUs without preference, pin or allowlist; no Windows.

Export only actor/critic parameters, fitted gain, terminal alpha/belief/reward scale and compact saved evaluation data from the source workers. Full checkpoints and replay remain remote; export JSONs allow idle-node scheduling. Return only compact results. No runtime optimization, new training or manuscript edits.

## Interpretation And Limitations

If discount reverses an intervention's cost ordering, investigate objective/evaluation alignment before changing the critic. If both cost definitions prefer the fitted first action while a frozen critic score prefers the actor action, investigate that score component and continuation mismatch. Neither pattern alone admits a training change. Q is a reward-normalized soft value for stochastic future actions, whereas this diagnostic uses deterministic continuation; disagreement is not a calibrated critic-error certificate. The pointwise entropy surrogate is not the expected actor objective. ID results have training-distribution support; OOD results are descriptive. A first-action effect cannot explain the full actor/reference gap by itself.
