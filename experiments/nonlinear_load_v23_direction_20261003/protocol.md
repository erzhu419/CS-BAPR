# Nonlinear load v23: seed18 frozen mean-direction diagnostic

Use v22 last seed18 floor/no-floor checkpoints t128679/t128680, whose pre-floor episode traces match. Exclude seed17's cross-device trajectory divergence from causal analysis. No training, fitting, new seeds, algorithm/environment change or manuscript edits; historical HOLD remains unchanged.

## Measurement

Both models run within one scheduler task on the same automatically selected Linux GPU. Evaluate radii0.2/1, severities1/50, paired paths60000--60099 and common initial states/external disturbances. For each model/state, the latent-mean direction is the unit vector from its current mean toward the fitted latent mean `-gain*observation`; hold its log standard deviation fixed. Record the distance. Perturb only the first latent mean by plus/minus0.01 and0.005 along this direction. Baseline plus four perturbed branches each use their own frozen deterministic actor at steps1--199:4,000 controller episodes total. Reproduce baseline v22 total/state/command costs at rtol1e-4/atol1e-3. Record raw/gamma0.99-discounted total cost, state/command and first50/last150 state cost. Compare central derivatives at both step sizes, including path-sign disagreement.

At the same initial states, compute local output-mean directional derivatives of Qmean, beta*Qstd and the native reparameterized entropy bonus. Beta=-2-2*terminal weighted_lambda. Use each checkpoint's frozen alpha, belief and reward scale, native unbiased ensemble std, tanh density correction/epsilon1e-6 and action range. Compare zero-noise critic directions separately from eight-draw Monte Carlo directions with shared CPU Torch normal draws92000--92007. Gradients of risk and soft scores are component sums. The Gaussian mean moves inside both the sampled action and its Normal density; this is not v21's fixed-action pointwise entropy score. No optimizer, belief or reward-EMA updates.

## Verification And Placement

Focused tests verify native reparameterized output-mean gradients, first-step-only changes, analytic finite-difference costs/discount index, zero direction, sign conventions, source reproduction and unchanged parameters. One CUDA preflight uses both models, radius1/load50, two paths and two noise draws (20 episodes). Require finite derivatives, cost conservation, baseline reproduction, exact paired external paths and zero updates before one formal task. Only compact actor/critic/gain/scalar exports and result JSONs return; checkpoints/replay stay on source workers. Scheduler has no node/GPU preference, pin or allowlist, and excludes Windows. No runtime optimization.

Study: `experiments/nonlinear_load_v23_direction_20261003`; results: `results/nonlinear_load_v23_direction_diagnostic`. Summarize with `python3 scripts/summarize_nonlinear_load_v23.py` using the frozen package runtime. No new performance/admission threshold is introduced.

Completed: preflight t128875 and formal t128876 exited0, with all frozen checks passed. Formal execution used automatically selected node007/GPU0 for both models and took116.1 seconds. See `docs/nonlinear_load_v23_direction_results.md`; historical HOLD is unchanged and no training intervention is admitted by this diagnostic.

## Interpretation And Limitations

Compare cost reduction directions against deterministic critic, risk and Monte Carlo soft-score directions to localize disagreement. Actual rollouts have deterministic continuation; critics were trained with reward-normalized stochastic soft min-ensemble targets. Agreement/disagreement is not a calibrated critic-error certificate. Eight draws approximate an expectation, and finite differences approximate a local derivative. These are conditional output-mean directions at fixed log std, not full actor parameter/Jacobian/Adam updates or a reconstruction of training. Only seed18 and two radii are examined; OOD results are descriptive. Findings guide the next diagnostic rather than admit another training variant.
