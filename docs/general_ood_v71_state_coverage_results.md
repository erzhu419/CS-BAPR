# v71: coverage fixes frozen gravity paths, not compound

t136916: node007 CPU, exit 0, 400.77 seconds. 96,000 exact true-mode replays;
576,000 simulator calls, zero optimizer updates. Pulled 233,647-byte scalar
JSON only. Both expanded banks use 18,600 labels per mode per fitting/calibration
role; the v69 null streams and all readout rules remain matched.

Fresh last64 mode accuracy (seeds 135120/121):

| Mode | v69 | Equal-count normal-only | Shifted-state |
| --- | --- | --- | --- |
| high_grav | 0% / 0% | 0% / 0% | 100% / 100% |
| compound | 0% / 0% | 0% / 0% | 12.5% / 0% |

Gravity first50 margins also became positive on both fresh seeds and both v70
replays. Shifted-state normal mean risk remained <=0.06061 across all groups.
The old updating-policy groups still failed the joint detection/identification
screen. Overall HOLD; do not connect belief or promote control claims.

Compound tail energy margins favor truth (+1.668/+1.538), but the global
Gaussian log-determinant term is -2.929, giving net -1.261/-1.390. Prediction
error/gap ratios improved to 1.742/1.629 but remain above 1. Next: freeze v71
forward models and compare global versus state-conditioned residual density,
retaining complete Gaussian normalization and consistent normal calibration.

## Limitations

One source seed, mild known profiles and privileged simulator labels. Density
support and fitting support change together. No closed-loop return evidence.
