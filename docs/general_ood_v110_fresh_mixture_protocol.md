# v110: frozen normal-mixture confirmation

Freeze v109 `normal_density_mixture` before new outcomes. Compare with v98
`diverse_recipe` and v106 `standard_mean_incumbent_covariance`. Use the exact
old/new mean banks, complete v98 covariance geometry and the three arms' v109
phase calibrations. No recalibration, model refit, weight search or candidate
selection from confirmation labels. Primary condition remains
`suffix_disagreement_mix`; retain support-only and trace-matched controls.

Fresh environment seeds: 135250/135251/135252; native smoke: 135249. Assert
separation from development, reference and calibration seeds before collection.
Each seed: five modes x frozen/updating, 6000 steps/path, switch at 3800.
Reuse v95's paired native collector, source training seed1 at update99745,
shared context/replay and fixed-minus2 continuation. Candidate readouts do not
control collection. Normal/changed prefixes match; recursive history persists
across resets. All three arms score the exact same trajectories.

Formal totals: 30 paths, 180,000 actual transitions, 900,000 shadow calls,
1,080,000 total environment calls and 90,000 optimizer updates. Scalar JSON
only returns locally. Check smoke's 90 path/arm/condition series and exact
production component likelihood agreement before dispatching formal shards.

Report frozen/updating and per-seed early/tail recognition, nonpositive margins,
matched departure contrasts, normal classification and maximum normal-window
mean risk. Judge replication jointly, not by demanding zero movement in every
normal window. Preserve adverse seeds. Any later redesign makes these seeds
development data. Actor integration remains a separate next decision.

## Limitations

New environment seeds do not replicate source training or other environments.
Reference budgets differ and mixture components share data. Mean risk is not
a false-positive rate; offline readout confirmation is not a return experiment.
