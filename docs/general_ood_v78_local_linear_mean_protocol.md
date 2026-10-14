# v78 Local Linear Residual Mean

Use the updating mixed bank's same 23,000 saved labels per mode and 64-neighbor
state/executed-action geometry. Replace local constant residual means with a
local affine Ridge prediction. Inputs remain in the frozen 14-dimensional
standardization; subtract the query input from its neighbors. Fit the paired
five-mode residual targets together using sklearn Ridge, mean-loss penalty
0.001 (alpha 64*0.001), and an unpenalized intercept. The intercept predicts the
query residual. Query next-state and mode labels never enter fitting.

Compare linear and unchanged constant means under each saved frozen/updating
conditional covariance. Keep frozen mean/frozen covariance as a fifth anchor.
Preserve global forward coefficients, scales, covariance estimation, recursive
hazard 0.02 and episode handling. Calibrate only the two new risk references
on the same saved v69 normal-null paths; do not tune ridge on replay results.

Replay all 65 paths as development evidence. Require three unchanged controls
to reproduce v77 scalar statistics. Measure mean prediction RMS, early/tail
identification, normal risk, historical regressions and density decomposition;
template-gap diagnostics remain unavailable on 17 historical paths.

Synthetic affine-recovery, density-equivalence, query-target independence and
resume tests precede a native saved-bank smoke and scheduler dispatch. No
simulation, RL optimizer updates, seed expansion or actor feedback. Only scalar
JSON/logs return locally; arrays and resumable progress remain server-side.
Fresh confirmation follows only if this mean intervention is supported.
