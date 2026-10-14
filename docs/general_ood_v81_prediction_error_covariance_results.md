# v81 Prediction-Error Covariance Results

Task `t137647` completed, exit 0, in 1,587.08s (26m 27s); scalar sync succeeded.
All 65 paths completed with the same 12,000 normal-null transitions. Five
reused controls exactly match v78; prediction RMS is unchanged on every path.
Result: `results/general_ood_v81_prediction_error_covariance/prediction-error-covariance/prediction_error_covariance.json`.

Compared with v78 updating covariance, v76 early/tail accuracy improves from
56.50%/85.55% to 60.75%/85.74%; v62 from 73.50%/62.89% to 82.25%/64.65%.
v74 becomes 95.75%/98.05% and passes the existing screen. v61 is mixed:
80.00%/49.61% becomes 76.00%/51.95%; heavy tail remains 15.62% and high_grav
tail remains 0%. Maximum normal mean risk coefficient is 0.02874, still <=0.1.

Decision: do not adopt pure LOO-error covariance; retain v78. Negative recursive
true-mode-margin windows increase from 14 to 18: two resolved, six introduced.
v76 early positive-margin pairs fall from 5/8 to 1/8; v73 loses its screen pass.
Accuracy gains do not imply better calibrated mode evidence. For v62 updating
seed134021 heavy tail, accuracy rises from 21.88% to 32.81%, while mean density
margin worsens from -15.92 to -131.04 (energy -133.50, volume +2.46).

The unchanged means establish that these changes come from covariance/readout
calibration. The severe negative energy margins are consistent with inadequate
query prediction-uncertainty estimates; they do not prove a universal cause.

Next: select a mixture of raw-residual and prediction-error covariance using
trajectory/block-held-out reference predictions across the known modes. Keep
v78 means fixed and select weights without these query outcomes; retain both
pure endpoints and normal/historical checks before fresh seeds or actor feedback.

Limitations: development replays, one source training seed and mild known modes,
not RL return evidence. Controls are reused, not independent replications.
