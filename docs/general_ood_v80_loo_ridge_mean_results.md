# v80 Reference-Only Ridge Selection Results

Task `t137617` completed, exit 0, in 3,100.28s (51m 40s); scalar sync succeeded.
All 65 paths completed; both new readouts used the same 12,000 normal-null
transitions. All five reused controls exactly match the saved v78 statistics.
Result: `results/general_ood_v80_loo_ridge_mean/loo-selection/loo_ridge_mean.json`.

Decision: do not adopt the LOO selector; retain v78 fixed-ridge means.
With updating covariance, v76 early/tail accuracy changes from 56.50%/85.55%
to 55.00%/85.35%; v61 from 80.00%/49.61% to 76.50%/49.61%. v62 is mixed:
73.50%/62.89% becomes 72.00%/63.48%. With frozen covariance, v76 accuracy
is unchanged at 64.50%/80.86%. v74 frozen paths retain 96.50%/100.00%.

The failed recursive true-mode-margin windows are exactly the same as v78:
15 with frozen covariance and 14 with updating covariance, with none resolved
or introduced. Normal mean risk coefficients remain <=0.1; maxima across
groups are 0.05250/0.00798 for frozen/updating covariance, not false-positive rates.

On v76 changed paths, alpha 0.064 is selected in 97.00% of early and 98.24%
of tail transitions; v61 fractions are 83.00%/96.48%. Local reference LOO
selection does not address the observed query failures. This is consistent
with measuring within-neighborhood fit rather than query extrapolation error.

Next: keep v78 means/alpha fixed and separately test covariance estimated
from cross-validated local prediction errors against the existing raw-residual
covariances. Use reference data only and retain matched controls; do not expand
the ridge grid, fresh seeds or actor feedback before this bounded test.

Limitations: development replays, one source training seed and mild known modes,
not RL return evidence. Controls are reused, not newly independent replications.
