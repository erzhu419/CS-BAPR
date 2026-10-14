# v100: equal-budget reference-age mean/support pilot

Reuse native v97 seeds135200/135201, keeping forward models, scales, 64-neighbor ridge and original v94 support fixed. Each recipe adds 2200 rows per seed (4400 total): diverse_suffix retains high_grav/compound steps3800:6000 at stride2; late_prefix takes normal steps1600:3800 at stride1; age_balanced combines that prefix at stride2 with both OOD suffixes at stride4. All rows retain five stored counterfactual next-state labels. Global coefficients and scaling are unchanged.

Compare all 60 first-post50/last64 windows from the 30 v95 paths, plus the two v62 tail controls: 62 windows, 3548 transitions. Reproduce incumbent prediction RMS before interpreting neighbor support, leave-one-out diagnostics, prediction errors and available oracle gaps. A four-window native smoke checks the new payload construction before the full mean pilot. No new simulation or optimizer updates, no error-bank fitting, covariance scoring, calibration or actor feedback.

## Limitations

Already inspected development paths, one source policy and known modes. Prefix-only rows are more temporally dense; the age-balanced recipe preserves two OOD sources. Mean improvements are not belief or policy gains. The historical v62 controls lack oracle observations.
