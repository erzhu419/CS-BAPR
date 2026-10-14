# v105 mean/support pilot

v104 finished eight collection paths with matched standard/exploratory budgets.
Each behavior supplies 4400 new reference inputs from seeds 135230/135231.

Append each collection only to the same v98 diverse mean reference (29600 to
34000 rows); keep the raw bank at 27400 rows and preserve forward coefficients
and scaling. The primary equal-budget contrast is exploratory vs standard,
each using 8800 extension source-index rows including the common v97 inputs.
The smaller incumbent is a descriptive comparator, not an equal-budget arm.

Analyze 62 fixed windows: first_post50/last64 for all 30 v95 paths, plus v62
seed134020 matched updating normal/slippery tails. Report the six prior coverage failures,
changed updating/frozen paths, and normal/historical controls separately.
Prediction error is mean pointwise coordinate RMS of observation-delta error
normalized by the frozen v72 target scale. Report 64-neighbor distances, new
reference neighbor share, per-source share and local leave-one-out diagnostics.

Verify incumbent predictions against cached v98 statistics for every window;
smoke also checks all three means against the production local-linear predictor.
No covariance computation/refit, calibration, posterior replay, actor change or
environment interaction. Only meaningful mean/support improvement warrants
subsequent recalibration and complete belief replay.

## Limitations

These are inspected development windows with simulator-known labels from one
source policy. Local reference diagnostics are not independent deployment error,
recognition accuracy or RL return evidence. Keep all arrays native; sync JSON only.
