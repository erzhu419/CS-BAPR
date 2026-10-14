# v98: partial coverage benefit; HOLD actor integration

Both replays completed successfully on node007 in about 2h19m. Each contains all 95 development paths and 570000 replayed transitions. Trajectory keys and unchanged v94 controls match between recipes. All 144 changed-window departure contrasts remain positive.

| Directional-disagreement reference | All changed windows / 144 | Updating windows / 64 | v95 updating / 24 | Non-positive v95 margins / 24 |
| --- | ---: | ---: | ---: | ---: |
| Unchanged v94 | 86.41% | 71.03% | 61.73% | 9 |
| Add normal/slippery inputs | 89.76% | 78.65% | 65.15% | 9 |
| Add high_grav/compound inputs | 91.10% | 81.39% | 68.12% | 7 |

Accuracy equally averages changed paths and the first-post50/last64 windows. Diverse references repair v95 seed135190 slippery/high_grav early posterior margins, but the other seven failures remain. Mean early prediction RMS is almost identical for the two recipes on v95 updating (0.21879 / 0.21890), so their accuracy difference is not evidence of uniformly better prediction means. The old recipe repairs v62 seed134020 slippery last64 whereas the diverse recipe does not; this is a matched-recipe tradeoff, not a new regression versus unchanged v94.

On v95 updating, the diverse recipe without disagreement scores 68.47%, versus 68.12% with disagreement. The largest normal-window mean risk changes 0.006888 -> 0.010140 / 0.010024 for old/diverse directional references. Keep the coverage benefit distinct from an added-disagreement benefit.

Next: a small native support and crossed mean/covariance attribution on the original nine failure/frozen pairs and the v62 recipe tradeoff with its normal control. No further collection or actor changes.

## Limitations

All paths are development evidence, conditional on one source policy and known mild simulator modes. Mean risk is not a false-positive rate. The two added-reference recipes have equal budgets; unchanged v94 has fewer rows/members. This is recognition evidence, not independent training replication or closed-loop return evidence.
