# v103: covariance regressions repaired, coverage failures remain

Task t140964 completed in 2h25m13s: 95 paths, 570000 transitions. Ordered trajectory keys, unchanged v94 controls and candidate prediction RMS match the comparators. Native preflight and matched 12000-transition normal calibration passed. Only scalar JSON was synchronized.

Primary suffix_disagreement_mix results:

| Metric | v98 | v101 | v103 |
| --- | ---: | ---: | ---: |
| v95 updating accuracy | 68.117% | 64.141% | 69.250% |
| All updating accuracy | 81.387% | 79.041% | 81.355% |
| v95 nonpositive-margin windows | 7 | 9 | 7 |
| Six previously unsupported windows, accuracy | 14.438% | 28.563% | 14.438% |
| v95 maximum normal-window mean risk | 0.004383 | 0.005172 | 0.004506 |

The three new v101 regressions recover to 96%, 92% and 73.438%; v98 gives 96%, 92% and 53.125%. V95 seed gains versus v98 are +1.055, -0.195 and +2.539 percentage points. All three covariance conditions modestly improve v95 accuracy, but the original seven negative-margin windows remain. Historical v62 slippery-tail accuracy drops from 32.813% to 31.250%.

Retain v98; HOLD actor integration. Next: stop prefix/suffix-ratio iteration and pilot broader reference behavior coverage with disjoint reference seeds, complete incumbent covariance frozen, and a matched distinct-input union budget. Verify mean/support gains before another full replay. No follow-up dispatched or paper tables changed.

## Limitations

One source policy, known modes and inspected development paths. Hybrid extension support uses 6600 distinct inputs versus 4400 per comparator. Normal mean risk is not a false-positive rate; recognition results do not establish RL-return gains.
