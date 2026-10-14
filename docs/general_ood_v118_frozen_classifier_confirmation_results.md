# v118 frozen confirmation

All three tasks (t141627/28/29) completed with exit 0; 30 paths, 180000 real
transitions and 90000 optimizer updates were merged. Only 1432482 bytes of
formal scalar JSON were transferred; arrays/checkpoints remain on node007.

Updating paths, averaged equally over the three environment seeds:

| Metric | Matched classifier | v98 |
| --- | ---: | ---: |
| Changed first-50 mode accuracy | 98.83% | 93.17% |
| Changed last-64 mode accuracy | 100.00% | 98.70% |
| Normal full mean risk | 0.00649 | 0.00229 |
| Maximum normal window mean risk | 0.02805 | 0.00306 |
| Normal signal-threshold exceedance | 9.13% | 6.98% |
| Minimum changed-minus-normal risk | 0.78520 | 0.79799 |

Early accuracy gains by seed are +0.5, +14.0 and +2.5 percentage points.
Much of the gain is slippery seed135261 (50% to 100%). High-gravity recognition
is not uniformly better: seed135260 is 94% versus 98%, and seed135262 is 98%
versus 100%. All 24 updating early/tail risk contrasts remain positive for
both readouts. Frozen-path early accuracy is 99.50% versus 98.83%.

The updating paired seed-level early-accuracy gain is +5.67 percentage points,
with a 95% Student-t interval [-12.43, +23.77]. Normal mean risk increases on
every seed, though its absolute magnitude is small; its paired mean difference
is +0.00420, interval [-0.00150, +0.00989]. These are environment-seed intervals
conditional on one source policy, not independent training replications.

The candidate recognizes modes better on average, but v98 already separates
changed from normal strongly. Both readouts give almost saturated changed-mode
risk, so mode recognition need not change the scalar actor penalty meaningfully.
Admit a small closed-loop diagnostic, without adopting the candidate or tuning
models/thresholds on these results. Compare the two frozen readouts with both
fixed-minus2 and fixed-minus3 to distinguish adaptive feedback from a stronger
constant penalty. Measure fixed-budget reward, final-policy return and normal
cost. Historical adverse seeds remain development evidence.

## Limitations

Three new environment seeds, one source policy, known mild Hopper modes. These
paths used fixed-minus2 collection; neither candidate actor feedback nor return
was tested. Empirical class weights are not calibrated physical posteriors, and
threshold exceedance is measured rather than guaranteed. Reference budgets
differ. The main paper tables remain unchanged.
