# v117 temporal replay: stronger change detection, not actor adoption

Formal t141605 completed on node007 CPU, exit 0, wall time 189.76 s.
All 125 complete histories (750000 transitions; 96 changed / 29 normal paths)
finished, and all 76 v116 score windows reproduced. No new model fits or
environment interactions. Only 3769295 bytes of scalar JSON returned locally.

Updating paths: 44 changed and 11 normal; equal-path averages.

| Readout | First-post50 mode accuracy | Last64 mode accuracy | Normal full accuracy |
|---|---:|---:|---:|
| Incumbent v98 | 71.32% | 91.80% | 98.93% |
| Original classifier on frozen neural means | 87.14% | 99.64% | 97.99% |
| Matched classifier on frozen neural means | 89.00% | 99.18% | 98.36% |
| Nearest mean | 54.09% | 99.96% | 99.78% |

Matched frozen-path early/tail accuracy is 99.27% / 100.00%. Across all paths,
the matched arm has positive changed-minus-normal risk in all 192 early/tail
windows (minimum mean-risk contrast 0.55492). Minimum changed first50 mean risk
is 0.55493; maximum normal mean risk over the existing normal windows is 0.03512.
This is useful departure evidence: the current continuation interface uses a
scalar risk to adjust beta, not the identity of the changed mode.

## Remaining Issues

Both compound failures persist: seed135190 first50 is 0% correct (all slippery),
seed135252 is 48%; their mean risks are nevertheless 0.873 / 0.850 and both
tails reach 100% recognition. Seed135192 high_grav recovers to 84% / 98.44%,
but still trails the original classifier's 96% / 100%. Other regressions remain:
seed135190 heavy falls from v98's 100% to 54% early accuracy. This is not uniform
mode-recognition dominance. Temporal accumulation can amplify biased evidence.

Matched updating normal threshold exceedance averages 10.60%, with 31.18% on
seed135192. That path's full mean risk is 0.02098; worst normal episode mean
risk is 0.07956. Low risk magnitudes do not establish calibrated false-alarm
rates. All paths remain inspected Hopper development evidence, conditional on
one source policy. Incumbent reference budgets differ. No return result exists.

## Next Step

Freeze the v115 final mean, v116 final classifier and v117 normal calibration.
Confirm unchanged readouts on three unused environment seeds, all five modes
and both frozen/updating policies, retaining normal, original-classifier,
nearest-mean and v98 controls. No tuning on confirmation outcomes. Assess
departure response and normal behavior jointly before a small actor-feedback
comparison against v98 and fixed-minus2. No further optimization of inspected
windows now. v98 remains retained; v118 is planned, not dispatched.
