# v42 Past-Only Dynamics Residual Probe

Frozen 2026-10-06 after v41 HOLD. Reuse the two v40 seed1 policies and the exact v41 paired Hopper protocol: 20 seeds 90000-90019, five modes, 1536 transitions, intervention before transition 1024, TimeLimit 1536. Actor/critics remain frozen, context updates normally, and no auxiliary signal changes actions. No RL optimizer updates or replay writes.

## Predictor And Calibration

Predict the actual state increment from state and executed action. Save the real next state before any reset, and separate physical termination from TimeLimit truncation. Do not create dynamics targets by differencing adjacent rows across resets.

For each policy/seed, fit once on normal-control transitions 0-511; calibrate on 512-767; leave 768-1023 untouched for validation. Apply those same frozen coefficients/scalers to all five cases. No validation or post-switch data select settings or update the predictor.

Fixed predictors: linear state-action, quadratic state-action, quadratic state-only. Quadratic features include all degree-one terms and upper-triangular pairwise products. Input means/stds and target increment stds come only from fitting data. Ridge minimizes mean squared standardized-target error plus 1e-3 times squared non-intercept coefficients. Input/target std floors are 1e-6. No sparsification, SINDy, NAU or oracle mode inputs.

Residual score is mean squared coordinate error divided by that coordinate's calibration residual mean square. Residual RMS floor is 1e-6 times fitting target std. Thus calibration mean score is approximately one, without implying that future null data are calibrated. The per-transition alarm reference is the calibration score's fixed 95th percentile. Neither floor nor quantile is selected from OOD observations.

Report validation256, first_post16, first_post256, second_post256: mean/median/p95 score, first-transition score, fraction above calibration95, nonterminal mean and termination/truncation counts. At transition 1024 the paired state/action is still identical; this contrast isolates the immediate physics response before trajectories diverge. Nonterminal aggregates can still include post-reset states. Also retain v41's previous-chunk reward/Q/L1 reference on the identical trajectories.

## Execution And Judgment

First operational CUDA preflight: seeds 99000-99001, normal/high_grav, both policies, 8 rollouts. Check causal fitting, true terminal targets, prefix pairing, finite statistics, source counters/config/runtime and exact v41 reward/end traces. If these fail, repair execution before formal submission. Do not choose predictors/settings from preflight signal performance.

Then 200 formal rollouts, 307200 interactions. Judge residual forecasting against untouched validation and no-change post windows. Poor null forecasts retain HOLD even if OOD scores are large. Compare both policies/profiles and early response separately from falls; a signal that only increases after failure is not adequate adaptation evidence. Predictor capacities are reported separately, not selected by OOD performance. A useful residual is a prerequisite for a later matched-budget belief-feedback test, not evidence of policy improvement.

Checkpoints remain on node007, so the probe is data-local there; scheduler chooses an idle GPU with no preferred/required GPU flag. No Windows. Sync only compact JSON/logs, never checkpoints or full trajectories.

## Limitations

One trained seed, two frozen policies, local per-trajectory predictors and repeated evaluation seeds. Deterministic policies couple actions to state/context, so the state-only comparison is not causal action identification. Limited pre-switch coverage can cause extrapolation error. This is neither an extreme50x control evaluation nor independent confirmation of revised CS-BAPR.
