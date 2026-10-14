# v78 Local Linear Mean Results

Task `t137307` completed, exit 0, in 3,121.66s (52m 2s); scalar sync succeeded.
All 65 paths completed; constant controls reproduced v77 statistics.
Result: `results/general_ood_v78_local_linear_mean/mean-correction/local_linear_mean.json`.

Local linear means improve matched v76 identification accuracy under both
unchanged covariance choices. Early/tail windows are first50/last64:

| Updating mean | Covariance | Early accuracy | Tail accuracy |
| --- | --- | --- | --- |
| Constant | Frozen | 50.50% | 70.12% |
| Linear | Frozen | 64.50% | 80.86% |
| Constant | Updating | 33.00% | 72.27% |
| Linear | Updating | 56.50% | 85.55% |

Linear/frozen preserves v74 frozen-path performance (96.50%/100.00%).
Linear/updating improves v61 from 48.00%/42.58% to 80.00%/49.61% and v62
from 58.75%/58.01% to 73.50%/62.89%. Both linear conditions retain positive
risk contrasts on every changed pair and normal mean risk coefficients <=0.1.

Decision: HOLD. Both v76 linear conditions still fail the identification screen.
With updating covariance, v76 slippery early accuracy is only 24%; v61 heavy
tail is 15.62% and high_grav tail is 0%. v76 compound early accuracy falls
from 48% to 46% (87% to 74% with frozen covariance). All four v76 mode-wise
early prediction RMS values increase despite aggregate accuracy gains;
all four v61 tail RMS values also increase. The fix is not uniformly better.

Next: inspect neighborhood support, ridge leverage, linear correction size and
covariance-weighted errors on the server in the failing windows. Extrapolation
is a hypothesis, not an established cause. Select a supported reference-only
correction before fresh confirmation or actor feedback.

Limitations: development replays, one source training seed and mild known modes;
these are belief diagnostics, not RL return evidence. Oracle template-gap
diagnostics cover 48 paths; density/belief attribution covers all 65.
