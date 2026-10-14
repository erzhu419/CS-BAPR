# v63 server-only readout attribution

Replay the12 v62 test traces and five v61 development traces using the frozen
v60 model and frozen v62 innovation reference. Require first50/tail signal and
risk to match the corresponding original observers within1e-8 before
interpreting the attribution. This round adds0 interactions and0 updates.

Report signal/threshold margins, instantaneous versus smoothed risk, innovation
dominance, paired11-coordinate residual displacement and state/action input
energy. The latter uses the existing dynamics input normalization, with a
descriptive95th percentile from the two saved v62 normal-fit traces.

At normal calibration age knots around first50 and tail, compare frozen v60
thresholds against the actual v62 normal95th-percentile values. Replay the old
v61 negatives separately; keep their counts separate from v62 test results.
All readouts remain causal and unchanged. Known boundaries only label offline
windows and paired diagnostics. No threshold sweep or parameter selection.

Use node007 CPU because all21 input traces and both references are already
there. Return only a small JSON/log; raw traces and checkpoints stay in place.
