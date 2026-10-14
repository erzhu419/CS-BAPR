# v45: Causal reset localization

Reproduce v44's frozen seed1 Hopper policies, quadratic action-conditioned
predictor and independent normal reference fit/calibration splits. Formal
replay uses the same 20 test seeds, normal and persistent heavy only: 24
reference plus 80 test rollouts, 196,608 interactions. Preflight uses the same
normal/high-gravity cases as v44 for exact source comparison. No RL, replay
writes, policy/context changes, new seeds or OOD parameter search.

Keep Student adaptive parameters, causal update timing and `log1p(score / q95)`
fixed. Compare episode-centered with already-fitted global-centered calibration;
neither refits on tests. For each calibration-group change, also predict the
current residual under the previous group's center/scale/threshold using the
same pre-observation belief/history. This one-observation counterfactual tests
whether normalization alone changes the reset decision; it is not a repaired
history or a proposed policy input.

Export all posterior-majority-reset indices, all calibration-boundary indices,
per-phase reset-peak indices and their +/-1-transition windows. Retain scalar
episode age, ending indicators, calibration score/threshold, actual and held-group
signals/posteriors, pre-observation mature predictive mean/variance, likelihood
contrast, age, lambda and target-scaled residual RMS. No raw state/action or
full trajectory export. Report transition counts as well as cases containing
an event; overlapping windows are not independent observations.

Require exact v44 model/reference/config/runtime, rewards/endings, residual
statistics and episode-centered belief statistics before causal interpretation.
Tests distinguish age boundaries from episode-reset boundaries, verify same-step
counterfactual timing and unchanged v44 replay, and retain all event indices.

Next repair depends on observed cause. Use normal-only evidence to repair
calibration or residual modeling; inspect persistent heavy without selecting
parameters from it. Multi-switch and matched-budget RL remain subsequent steps.
