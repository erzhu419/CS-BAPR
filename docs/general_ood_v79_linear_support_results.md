# v79 Linear Support Results

Native smoke `t137613` passed in 14.51s (228 transitions). Attribution task
`t137614` completed, exit 0, in 85.56s; scalar sync succeeded. All 65 paths,
130 windows and 7,410 transitions completed. Prediction RMS and density
statistics reproduced v78. No reference, readout or actor changed.
Result: `results/general_ood_v79_linear_support/attribution/linear_support.json`.

There are two remaining failure patterns, not one universal correction:

- v76 seed135180 early slippery/compound have mean query leverage 1.48/1.35
  and prediction-weight L1 norms 7.18/6.93. Linear RMS rises from 0.2276/0.1855
  to 0.4393/0.4521. Under updating covariance, energy gains are -12.42/-14.95:
  the correction penalties exceed the alignment benefits. Their tails improve.
- Misranking also persists with smaller leverage and beneficial corrections.
  v62 updating seed134021 heavy tail has leverage 0.46 and energy gain +5.31,
  but its high_grav margin remains -15.92 (energy -18.28, volume +2.36).
  v76 seed135181 slippery early has leverage 0.34, RMS improves to 0.1485,
  yet recursive accuracy is 34%; tail accuracy is 51.56% despite RMS 0.1445.

v61 heavy tail also improves weighted energy (+8.39), but RMS increases
from 0.1598 to 0.1903 and recursive accuracy remains 15.62%. Neighbor support
and unweighted prediction error alone cannot determine correct mode ranking.

Next: test reference-only ridge selection using local leave-one-out errors,
with one shared regularization strength across modes, no query next-state or
true-mode label in selection, and both covariance banks unchanged. Retain
v78 linear and constant controls, recalibrate on the same normal-null data and
replay all 65 paths. This targets early regressions; persistent ranking errors
would then justify a separate prediction-error covariance intervention.

Limitations: development attribution, one source training seed and mild known
modes, not RL return evidence. Leverage/weight associations are not causal
proof; local reference leave-one-out errors are not deployment validation.
