# v66 covariance mixing results

t136637 completed on node007 CPU, exit0, in132 seconds. Seven frozen-runtime
tests passed. Added RL interactions/updates0;17 cells replayed102000 saved
transitions. Only542746 bytes of scalar JSON returned. All inherited controls,
paired contrasts, budgets and gate summaries match the results.

Six coupled normal-only holdouts selected global covariance weight0.9379491.
Normal validation NLL is-20.673977, versus global-only-20.341242 and
local-only-19.156378. Separate normal calibration set score/age thresholds.

Both groups pass normal bounds: largest means0.086924 (v62) and0.038879 (v61
replay). Tail contrasts pass8/8 and4/4. Early contrasts pass7/8 and3/4, so the
main candidate remains HOLD. Relative to v65's local-covariance main, normal
and tails improve but early counts decrease from8/8 and4/4. Relative to its
center-only ablation, one additional early contrast becomes positive.

Remaining failures are gravity seed134020 first50 (zero contrast, all signals
below threshold) and gravity seed132020 first50 (-0.008966, normal transient
slightly stronger). Both have positive50-100 contrasts (+0.005047/+0.083776).

Next: retain the v66 level/tail channel and test normal-only causal residual
prediction innovations to distinguish normal transients from changes. Keep
fit/calibration separate, all controls and the first50 gate unchanged; no
oracle boundary or new RL-seed expansion.

## Limitations

Reused development trajectories from one source training seed. Positive
50-100 contrasts do not satisfy the first50 gate. Normal predictive likelihood
and this fixed-beta replay establish neither independent confirmation nor
adaptive RL gains. Full references and raw data remain server-side.
