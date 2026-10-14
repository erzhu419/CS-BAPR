# v74 Density Coverage Results

Task `t136994` completed on node007 with exit code 0 in 1h 05m 51s.
Source: `results/general_ood_v74_density_coverage/screen/density_coverage.json`.
The run used 84,000 actual transitions, 420,000 counterfactual steps and no
optimizer updates; all 84,000 exact true-mode replay checks passed.

## Result

Fresh frozen-policy tests use seeds 135160/135161 and four changed modes.
Early means the first 50 post-switch transitions; tail means the last 64.
All values below are recursive mode-identification accuracies.

| Density bank | Early mean | Tail mean |
| --- | ---: | ---: |
| Original v72 | 83.25% | 91.21% |
| Equal-label normal-state extension | 79.75% | 88.87% |
| Equal-label slippery-state extension | 92.00% | 99.02% |

The slippery extension passes the fresh sign-based screen (8/8 positive true-mode
margins in both windows), as well as all frozen development replay groups.
Its fresh normal mean risk coefficient is at most 0.004053 across evaluated
windows; this is a continuous coefficient, not a false-positive rate.
Slippery early/tail accuracy improves from 53.00%/70.31% to 97.00%/99.22%;
compound improves from 95.00%/96.09% to 100.00%/99.22%.
The equal-label control supports state-coverage specificity rather than a
generic benefit from adding labels; density means and covariances change together.

## Limitations

Heavy early accuracy falls from 86% to 78%, and high-gravity early accuracy from
99% to 93%. Historical updating replay groups still fail: v62 early/tail means
are 50.75%/58.79%, and v61 means are 39.50%/42.19%. Overall decision remains
`hold-density-coverage`. This is observer evidence from one trained source seed,
mild known modes and privileged offline counterfactual labels, not a return
comparison or controller benefit. Previously inspected trajectories are development data.

## Next Step

Compare reference-neighborhood support and clipped velocity observations on
server-resident updating versus frozen traces. Distinguish a coverage gap from
observation aliasing before choosing updating-state calibration or an observation
change. Keep model arrays and traces on the server; return only scalar summaries.
No controller integration, seed expansion or hazard tuning in this step.
