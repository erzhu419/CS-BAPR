# v96: HOLD; prioritize policy-matched reference coverage

18 paired policy windows, 956 transitions, 54 reproduced v95 attributions; four local tests pass. No simulation or optimizer updates.

All nine updating failure windows have poorer state/action support and over 99% of neighbors from one reference seed. Equal-window nearest radius: 0.388 frozen versus 0.888 updating (2.29x).

Local-constant diagnostic: updating prediction RMS 0.328 -> 0.285, but instantaneous accuracy 17.18% -> 14.83%; all nine density margins remain negative. Keep the linear mean and HOLD actor integration.

Next (planned, not dispatched): policy-matched reference collection versus the old collection procedure at equal budgets. Preserve model/covariance forms and calibration protocol, recomputing reference errors and phases. Screen existing development paths and normal controls before fresh confirmation.

## Limitations

Failure-selected, one-policy diagnostics are not causal or return evidence. Window-moment directions supersede unstable pointwise ratios on roundoff-sized oracle gaps.

Final scalar artifact: `results/general_ood_v96_mean_support/attribution_direction/mean_support.json`; native CPU task `t139955`, node007. Reference banks, trajectories and checkpoints remain server-side.
