# v96: paired mean/support attribution

Analyze the nine negative updating windows already observed in v95, paired with frozen-policy windows at the same environment seed, mode and time. Seven first-post-50 and two last-64 windows give 18 policy windows and 956 transitions. The native smoke uses the first pair (100 transitions).

Reuse v94's unchanged full-suffix mean, source-excluded error bank, covariance controls and phases. First reproduce v95's saved instantaneous prediction error, density margin, energy/volume decomposition and oracle-gap statistics. Reuse v79's analytic local ridge weights and local leave-one-out residuals to inspect radius, state/action distance, extrapolation leverage and reference-seed composition. Report window-level signed prediction-error projections onto all four oracle mode gaps in normalized target units: projection is sum(error dot gap) / sum(gap squared), with window RMS ratios and flattened-window cosines. Report nonzero-gap fractions, minimum gap RMS and window gap RMS separately. These moments replace pointwise ratios that were unstable on the actual roundoff-sized oracle gaps in seed135191.

Local-constant and global-forward means are diagnostic comparisons under the candidate's unchanged covariance, not new candidates. No simulation, optimizer updates, fitting, calibration, coefficient changes or actor integration. Only scalar JSON is synced; traces, reference banks and arrays stay on node007, where these inputs reside. No Windows placement or soft node/GPU preference.

## Limitations

These failure-selected windows are development evidence, not fresh confirmation. Frozen and updating paths share initial seed and schedule, not post-update states/actions. Support statistics do not prove causation; query-neighborhood leave-one-out errors are not held-out deployment errors. Covariance stays fixed for the mean comparisons. All averages weight windows equally. A window with entirely zero gaps has undefined direction/ratio (null); a merely nonzero gap does not establish practical identifiability.
