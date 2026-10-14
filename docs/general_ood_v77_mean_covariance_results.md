# v77 Mean/Covariance Results

Task `t137295` completed, exit 0, in 2,780.84s (46m 21s); scalar sync succeeded.
All 65 paths completed and both diagonal statistics reproduced v76.
Result: `results/general_ood_v77_mean_covariance/attribution-final/mean_covariance_attribution.json`.

On v76 development paths, early/tail accuracy is 36.50%/67.77% for frozen
mean/frozen covariance, 50.50%/70.12% for updating mean/frozen covariance,
37.00%/71.29% for frozen mean/updating covariance and 33.00%/72.27% for both
updating. All four fail the existing identification screen.

Updating mean/frozen covariance preserves v74 frozen-path accuracy
(92.00%/98.05%), but v61 accuracy falls to 28.00%/29.69% versus 48.00%/42.58%
with both updating. There is no universally successful moment swap.

On v76 slippery paths, updating-mean prediction RMS is 0.2519 early and
0.3036 tail versus nearest observation-template gaps 0.1001/0.1098; prediction
error exceeds the gap on 99%/100% of transitions. The early slippery/compound
margin with frozen covariance has energy contribution -6.353 and volume
contribution -0.793. Log determinant alone does not explain the confusion.

Next: replace updating-bank local constant residual means with 64-neighbor
local linear ridge predictions. Hold each frozen/updating covariance fixed;
retain both matched constant-mean controls and the frozen/frozen anchor.
Replay all paths before any fresh confirmation or actor feedback.

Limitations: development replays, one source training seed and mild known
modes, not RL return evidence. Template-gap diagnostics exist on 48 paths;
the other 17 retain density/belief attribution. Prediction RMS versus gap is
a diagnostic scale comparison, not a sufficient classification criterion.
