# v117 frozen-score temporal replay

Freeze the formal v115 ten-source neural mean and formal v116/v113 final
classifiers. Compare matched classifier, original classifier and nearest mean
on identical real transitions. No fitting, oracle query scores or score tuning.

Calibrate each arm on the same two v69 normal-null paths (12000 transitions).
Reuse the v73 recursive filter: hazard .02, uniform redraw, uniform initial
weights, posterior persists across episode resets. Preserve phase-age quantile
calibration and the 16-step departure readout. Only score inputs and resulting
normal thresholds differ. These are empirical recursive score weights, not
physical-density Bayesian posteriors; no temperature or hazard search.

Replay all 95 v98 development paths and all 30 v110 paths from step zero:
125 paths, 750000 real transitions, three score arms per path. Keep every mode,
updating/frozen path and normal control. Copy all three incumbent v98 condition
summaries from prior verified replays; primary comparator is
suffix_disagreement_mix. Reproduce all 76 v116 instantaneous score windows.

Native smoke uses the same formal models, full normal calibration and ten full
histories, including both negative-margin compound cases, seed135192 high_grav,
their normal controls and fresh frozen counterparts. Check full-history prefix
agreement before formal submission. Tests check real-transition dependence,
normal-only calibration equivalence and persistent causal state across resets.

Report early/tail recognition separately, normal full recognition, normal
signal-threshold exceedance fraction, normal risk means and worst episode mean
risk. Retain per-path windows and normal-episode scalars. A positive average
score margin does not guarantee temporal recovery. No new data, seed expansion,
actor changes, return claims, adoption or manuscript-table updates.

## Limitations

All paths are inspected development data from one source policy and known mild
Hopper profiles. Calibration data are reused. Exceedance fractions are observed
diagnostics, not guaranteed false-positive rates. Incumbent model/reference
budgets differ. Arrays and model states stay on the server; pull scalar JSON only.
