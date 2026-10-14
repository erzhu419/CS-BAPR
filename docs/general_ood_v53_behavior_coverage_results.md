# v53: normal coverage repaired, gravity separation remains weak

t135965 completed9 cells,10800 interactions/8400 updates in189.45 s, exit0.
Splits, budgets, paired prefixes and contrasts match; scalar JSON29184 bytes.

Held-out learning-normal mean departure fell from .910606 to .029529 (96.76%);
last64 from .996949 to .002174. Beta stayed-2; the readout did not affect actions.
Reference fitting/calibration remained normal-only, separate from testing.

Broad-reference change-minus-normal coefficients:

| Profile | First post50 | Last64 | All post400 |
| --- | ---: | ---: | ---: |
| heavy | .054277 | .003241 | .051083 |
| slippery | .270353 | .152146 | .213683 |
| high_grav | .003098 | -.000221 | .036335 |
| compound | .280079 | .128431 | .199430 |

Normal-null criterion passed; gravity tail separation did not. Formal remains
HOLD. Its tail signal mean rose from .086080 to .114062 despite the readout reversal.

t136164 exactly reproduced the gravity cell with1200 additional interactions/
updates in95.94 s; scalar JSON13755 bytes. No refit or readout change. Mean squared
causal16-step signed residuals rose from .015173 to .031494 in first50 and from
.039857 to .054953 in last64. This supports investigating a normal-calibrated
multivariate window readout on untouched seeds, not expanding adaptive RL yet.

## Limitations

One source/continuation seed, plus a development replay; no RL return confirmation
or50x OOD comparison. The tiny negative tail is not a significant negative effect.
Signed window energy is uncalibrated here. Five focused tests passed.
