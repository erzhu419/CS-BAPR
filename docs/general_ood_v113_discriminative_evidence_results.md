# v113: normal recognition improves, changed-mode controls regress

Formal `t141543` completed, exit 0, in 246.58 seconds. Both fixed classifiers
converged (validation/final: 719/756 iterations). All 76 windows / 4,248
transitions completed, with 228 expanded-mean RMS comparisons. The 684 copied
Gaussian control statistics match v112; their likelihoods were not recomputed.
Only 676,500 bytes of scalar JSON returned.

The fixed two-source validation accuracy is 89.86%, versus 86.62% for nearest
mean; source135230/135231 accuracies are 91.66%/88.06%. Neither validation
source enters classifier training, scaler fitting or training local-mean fits.
This validates the reference split, not the inspected evaluation outcomes.

Equal-window mean **instantaneous** accuracy; Gaussian condition is
`suffix_disagreement_mix`, not complete-history belief accuracy:

| Inspected population | v98 | v112 matched errors | v113 classifier | Nearest mean |
| --- | ---: | ---: | ---: | ---: |
| Current four failed updating windows | 8.00% | 9.00% | 15.50% | 20.00% |
| Current six normal controls | 94.33% | 89.00% | 99.67% | 99.67% |
| Current four matched frozen controls | 86.50% | 87.00% | 75.50% | 59.50% |
| Earlier six failed windows | 20.51% | 73.06% | 54.23% | 27.84% |

Seed135252 updating normal improves from v98/v112 68%/36% to 100%. Its
slippery/high_grav/compound windows remain at 10%/8%/6%, predicting normal on
82%/88%/86% of steps. High_grav nearest-mean accuracy is 46%, versus learned
8%; better reference validation does not establish the better shifted readout.
Across all normal windows, five improve versus v98 and none regress. Across
28 frozen changed windows, 18 regress and none improve; the worst is
seed135191 slippery first50, 80% -> 34%.

**Decision:** HOLD; retain v98, without full temporal replay, actor adoption or
seed expansion. Next proposed native diagnostic: freeze v113 weights, split
true-vs-competitor logits into intercept/linear/squared contributions, and
compare fitted versus simulator-correct centers on failures, normal controls
and the largest frozen regressions. This diagnostic is not yet dispatched.

**Limitations:** known-mode reference labels, one fixed two-source validation
split and previously inspected development windows. Nearest mean has a fixed,
untuned score scale; its NLL is not a matched probability-calibration baseline.
Normal instantaneous accuracy is not a calibrated false-alarm or return result.
