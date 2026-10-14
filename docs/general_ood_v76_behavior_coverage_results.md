# v76 Behavior Coverage Results

Task `t137090` completed on node007, exit 0, in 6,953.71s (1h 55m 54s).
All 18 new collection paths and 65 evaluation paths completed; result sync
succeeded. Budget: 108,000 actual transitions, 540,000 counterfactual steps,
108,000 exact true-mode replay checks and 84,000 optimizer updates.
Scalar result: `results/general_ood_v76_behavior_coverage/screen/behavior_coverage.json`.
Decision: **HOLD** (`hold-behavior-coverage`).

## Findings

Fresh updating tests, averaged over two seeds and four changed modes:

| Reference | First 50 accuracy | Last 64 accuracy | Maximum normal mean risk coefficient |
| --- | ---: | ---: | ---: |
| Original v72 | 36.25% | 58.59% | 0.05817 |
| v74 slippery | 36.50% | 67.38% | 0.03529 |
| Matched frozen mixed | 36.50% | 67.77% | 0.09877 |
| Matched updating mixed | 33.00% | 72.27% | 0.00633 |

Normal maxima use first800/full/last1000/last64; these are risk coefficients,
not false-positive rates. All fresh screens fail the existing mode-margin rule.

Updating versus frozen mixed coverage lowers early mean 64-neighbor radius
from 1.8309 to 1.5336 and raises added-neighbor share from 0.57% to 51.41%.
Coverage improves but early accuracy falls. Early heavy accuracy is 37% -> 32%,
high_grav 55% -> 33%, slippery 16% -> 19%, compound 38% -> 48%.
High_grav's normal-prediction share rises from 45% to 66%; heavy still mainly
confuses with high_grav. Early instantaneous accuracy also falls, 39.75% ->
37.50%, so recursive persistence is not the sole explanation.

Historical v61/v62 updating screens still fail. On v74 frozen development
paths, early/tail accuracy falls from 92%/98.44% to 86.25%/92.38%.

## Next Step

Use saved server traces for a 2x2 frozen/updating conditional-mean/covariance
replay. Split true-versus-competing likelihood margins into error-energy and
log-determinant terms, retaining fixed forward models, scales and temporal
rules. Recalibrate hybrid risk thresholds only on the saved normal-null paths.
This will guide mean or covariance changes; no new collection is needed yet.

## Limitations

One source training seed, two fresh test seeds, mild known modes and privileged
offline labels. Mean and covariance changed together; coverage alone is not a
complete explanation, but remaining coverage error is not ruled out. Replayed
paths are development evidence. The observer has no actor feedback, so these
results do not establish RL return gains or general OOD robustness.
