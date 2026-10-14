# v55 preflight: learning-matched age calibration

Task t136459 completed with exit 0. Scalar result: `results/general_ood_v55_learning_calibration/preflight/learning_calibration.json` (46,792 bytes). All 14 role/seed cells, exact paired prefixes, budgets and contrast calculations match the frozen protocol: 16,800 interactions and 15,600 updates.

The candidate passes the predeclared normal full/last64 mean <=0.10 and every changed-profile/seed first50/last64 contrast >0. Normal full means are 0.013891 and 0.004468; last64 means are 0 and 0.000443. Changed-minus-normal coefficients:

| Seed | Profile | Old first50 | New first50 | Old last64 | New last64 |
| --- | --- | ---: | ---: | ---: | ---: |
| 98620 | heavy | 0.071423 | 0.292147 | 0.148930 | 0.251042 |
| 98620 | slippery | 0.078195 | 0.269629 | 0.307791 | 0.384915 |
| 98620 | high_grav | 0 | 0.049625 | 0.000678 | 0.011362 |
| 98620 | compound | 0.063906 | 0.238120 | 0.308789 | 0.387347 |
| 98621 | heavy | 0.029006 | 0.265971 | 0.384079 | 0.454124 |
| 98621 | slippery | 0.237665 | 0.408695 | 0.427579 | 0.495834 |
| 98621 | high_grav | 0 | 0.067466 | 0.050508 | 0.154200 |
| 98621 | compound | 0.105015 | 0.325436 | 0.179944 | 0.278311 |

The old calibration replay reproduces its age thresholds within 1e-12. At ages 800/816/832/848, pooled thresholds exceed the updating-normal producer's maximum. On identical held-out trajectories, changing only the age-calibration population restores both early gravity contrasts. This supports calibration-population mismatch as a contributor to the earlier insensitivity; v54 remains HOLD.

## Formal
Task t136466 completed with exit 0. All 22 cells, paired prefixes and scalar contrast calculations match the protocol: 26,400 interactions and updates. Normal full means are 0.004931/0.004971/0.006561/0.002758; all last64 means are zero. Early contrasts pass 16/16; tails pass 15/16, so the frozen decision is HOLD.

Gravity first50 contrasts for seeds98720--98723 are 0.047948/0.045698/0.041344/0.079742. Tail contrasts are 0.000545/0.001511/0.000024/0. At seed98723 the tail signal is 0.121628 versus normal 0.057429, but both coefficients are zero. Age160/176 normal thresholds are 0.572946/0.602048 (preflight: 0.150452/0.247993); pooling initial and post-reset learning trajectories is a plausible contributor, not yet established.

## Next
Replay normal calibration98700/1 and failed gravity98723, reusing its saved normal control. Reproduce the old thresholds and recorded readouts, then separate initial/post-reset threshold populations and report signal/threshold/exceedance statistics. No candidate refit or changed gate; 3,600 added interactions/updates, server-only raw data. See the v56 protocol.

## Limitations
These are conditional detector diagnostics from one source-training seed at fixed beta=-2. Neither readout controls learning or actions. Formal gravity tails are zero or very small, despite positive early contrasts. This is not adaptive RL or 50x OOD return evidence.
