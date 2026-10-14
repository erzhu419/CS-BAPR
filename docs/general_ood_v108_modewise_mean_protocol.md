# v108 mode-wise mean ablation

v107 attributes the seed135192 normal regression to changed likelihoods, not
calibration. Compare four fixed density assemblies: old, new, old normal plus
new OOD, and new normal plus old OOD. Selection is by hypothesis column, never
by the trajectory's true mode. Reuse normalized mode-density columns; v106
already verified unchanged covariance, raw density and mixture weight.

Each assembly receives its own unchanged production calibration procedure on
the same 12,000 normal-null transitions (seed135060, frozen and updating).
Reproduce the six saved old/new calibrations and both original replay controls.
Replay all 95 development paths / 570,000 cached transitions, preserving full
belief histories across episode boundaries, in all three covariance conditions.
Smoke uses the four seed135192 normal/compound and frozen/updating paths.

Report original population/window metrics, six old failures, historical slippery
tail, normal episode risks/accuracy, and the normal-versus-compound log-density
changes. No new environment interaction, density/geometry fitting, actor updates,
threshold tuning or seed expansion. Only scalar JSON is returned locally.

## Limitations

One source policy, known modes and development-only paths. Mean reference budgets
differ by mode; this is not a matched total-budget comparison. Calibration is
reused-population, not independent confirmation. No actor or return claim.
