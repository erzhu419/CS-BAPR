# v94 Matched suffix uncertainty replay

Replay the same 65 saved paths / 96 changed early/tail windows after the v93 suffix reference NLL improvement. Keep the complete 25,200-row suffix mean, original 23,000-row raw component, weight 0.24619454443573643 and temporal rules. Before any query, compute full-bank errors with each source seed excluded from all original/added predictor rows. Deployment uses six source-deletion members; reference validation used five after outer exclusion.

Three paired conditions: `suffix_support_mix` (no added disagreement), `suffix_disagreement_mix` (`Sigma + D`) and `suffix_trace_match_mix` (`(1 + trace(D)/trace(Sigma))*Sigma`). Coefficient one is fixed. All share the same mean, base covariance and computed `D`; directional and rescaled covariances have identical traces. No query next state or true-mode label enters geometry. Reuse five v90 controls: v78 raw, v83 error/mix, v89 prefix/suffix frozen-error means.

Calibrate all three on the same two normal-null paths / 12,000 transitions using the unchanged recursive reference implementation, hazard, prior and reset rules. Compute paired likelihood arrays once per path and cache them server-side. Resume uses complete per-path progress; no truncation of the history before early/tail windows. Compare negative recursive windows, equal-window accuracy, normal risk, departure contrasts and fixed-pair attribution. Preserve slippery/compound and v61/v62/v76 counterexamples.

Native smoke checks the same three v90 paths and six windows: v76 seed135180 compound and v61 slippery/heavy. Require full-bank error pairing/exclusion, six-member correspondence, unchanged means, covariance formula, trace match, physical Gaussian scores and reproduced v83/v90 attribution; inventory all 65 paths and both nulls. Then dispatch the full replay. HOLD remains until result assessment; no actor feedback, fresh seeds, simulations, coefficient tuning or manuscript edits.

Use a fresh frozen runtime on node007 CPU, where existing inputs reside. Only scalar JSON/logs return locally; full-bank errors, likelihood arrays and phase curves stay in excluded `run_checkpoints`.

## Limitations
Reused development paths and one source training seed are not independent confirmation or RL returns. Unequal seed deletion measures sensitivity, not calibrated variance. Matching trace does not match determinant or precision; those differences are part of the shape comparison.
