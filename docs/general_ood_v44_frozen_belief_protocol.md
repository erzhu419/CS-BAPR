# v44: Frozen belief replay

Use the two v40 seed1 Hopper policies, independent normal reference splits, and
matched v43 development trajectories. No RL updates or changes to actions,
context inference, replay, mode profiles, or paper claims.

## Fixed Design

Pre-select v43 quadratic-state-action dynamics and episode-centered residual
calibration. Recollect the same reference/test seeds; fit/calibrate once per
policy. Input to all filters is `log1p(score / calibration95)` on every actual
transition. Calibration uses current episode age known from past endings.
Belief is initialized uniformly per rollout and is not reset on episode ends.

All filters use 20 capped run-length slots, hazard .02, and fresh predictive
variance 1. Reset is before the current observation, matching the existing
tracker. Compare these fixed likelihoods:

- `half-normal-fixed`: current zero-location half-normal with variance
  `1 / (1 + .05 * age)`.
- `student-fixed`: zero-location Student-t with four degrees of freedom and
  the same age-dependent variance; distinguishes heavier tails from learning.
- `student-adaptive`: Gaussian likelihood with normal-inverse-gamma prior
  `(mu, kappa, alpha, beta) = (0, 1, 2, .5)`, marginalized to a Student-t.
  A candidate age `r` predicts using the preceding `min(r, available_history)`
  observations. Update history only after scoring the current observation.
  The capped slot uses the last 19 samples, pooling longer paths. Learned
  predictive variance replaces artificial `.05` age narrowing in this arm.

Map each posterior to `lambda = E[exp(-.1 * age)]` and hypothetical
`beta = -2 - lambda`. Neither is passed to the agent. No OOD-based parameter
search or actor training follows automatically from operational preflight.

## Measurements

Keep v43 validation256, first_post16, first_post256, second_post256 windows.
Also freeze early_post64 [1024,1088), late_first64 [1216,1280), and
late_second64 [1472,1536). Save per-window mean/peak/end reset probability,
effective age, lambda, beta, and entropy; 16 post-switch 32-step bin means;
signal level; and physical termination/truncation counts. Compare to the
matched no-change rollout using seed-level contrasts. A posterior reset
probability above .5 is reported descriptively, not a tuned decision gate.

Before interpreting results, require v43-identical checkpoint/config/runtime,
reference metadata, selected predictor metadata, residual-score summaries,
and reward/termination/truncation traces. A mismatch stops interpretation.
Unit tests cover predictive formulas, causal update order, cap/reset behavior,
unchanged half-normal control, and recovery after a constant synthetic shift.

## Interpretation

The useful behavior is early switch response without comparable sham response,
followed by age recovery while a stable shifted signal remains elevated. If
adaptive prediction still repeatedly favors reset, retain that negative result
and inspect normal-conditional variation before any RL feedback integration.

Run-specific predictive likelihoods and posterior sufficient-statistic updates
are motivated by [Adams and MacKay (2007)](https://arxiv.org/abs/0710.3742).
This project's reset-before-current timing and finite recent-history cap are
explicitly different from their full Bayesian online changepoint detection.

## Limitations

This is reused development evidence from two frozen seed1 policies, not RL
performance or independent confirmation. The Student adaptive arm learns both
location and variance. Its real-valued predictive model approximates positive
log scores. The 19-sample history and coarse episode-age calibration may not
capture periodic locomotion or all normal regimes; later windows may include
physical failures. Preserve all three arms and all mode results.
