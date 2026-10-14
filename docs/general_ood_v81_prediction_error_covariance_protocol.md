# v81 Prediction-Error Covariance

Keep v78 updating-bank 64-neighbor linear means and alpha 0.064 unchanged.
Change only covariance: hold each reference row out of the same local fit,
then use its prediction error rather than raw forward residuals as OAS input.
All five mode vectors share the same neighbors and fixed regularization.

For the centered local design X, ridge hat diagonals are
h_i = 1/64 + x_i' (X'X + 0.064 I)^(-1) x_i.
Reference LOO errors are fitted residuals divided by (1-h_i), equivalent to
explicit row-held-out refits with the same alpha and neighborhood. Estimate
each 11-dimensional error second moment with OAS assume_centered=True and
add the existing 0.001*trace(OAS)/11*I ridge. Do not recenter prediction errors,
change the mean, use query next-state/mode in geometry, or inflate by leverage.

One new readout, linear_mean_loo_error_cov, retains all five saved v78 controls,
including fixed means with frozen/updating raw-residual covariance. Recalibrate
on the same 12,000 normal-null transitions and replay all 65 development paths
with unchanged history, hazard and reset/age rules. Verify prediction RMS
against v78 on every path; record covariance trace, shrinkage and LOO RMS.

Gate full replay on explicit-refit/OAS/physical-Gaussian tests and native smoke
on the same two failure paths (228 transitions), including unchanged means and
batch/readout agreement. Resume completed path prefixes only; caches stay remote.
No new simulation, seeds, actor feedback, leverage multiplier or paper-table
changes. Retrieve scalar JSON/logs only; no Windows jobs or machine preference.
