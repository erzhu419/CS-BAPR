# v95 Fresh environment-seed confirmation

Tasks `t139551`/`t139552`/`t139553` completed on node007 GPU0/1/2, exit 0, in 35m 04s / 34m 01s / 35m 14s. All three results synced: 30 paths and 48 changed windows. Frozen reference metadata matches across shards. Only 1,068,636 bytes of scalar results returned; budgets and detailed comparisons are recorded in the dispatch manifest.

| Frozen readout | Non-positive margins / 48 | Overall accuracy | Frozen-policy accuracy / 24 windows | Updating-policy accuracy / 24 windows | Maximum normal-window mean risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| v78 raw covariance | 15 | 74.84% | 97.16% | 52.52% | 0.007362 |
| Suffix matched support, no disagreement | 9 | 80.77% | 98.05% | 63.48% | 0.006894 |
| Suffix directional disagreement | 9 | 80.41% | 99.10% | 61.73% | 0.006888 |
| Suffix trace-matched scaling | 9 | 79.08% | 98.39% | 59.78% | 0.004610 |

All suffix conditions have the same nine negative windows, all under updating; none of v94's additional repairs replicate. Directional accuracy improves/regresses/is unchanged in 10/9/29 windows versus no disagreement. It beats trace scaling by 1.3307 points but loses 0.3516 points to no disagreement. Its gain over v78 therefore cannot establish an added-disagreement benefit. All 48 departure-risk contrasts stay positive. Frozen-policy screens pass at all three seeds; updating screens fail at all three.

Updating early accuracy falls 45.83% -> 42.33% with disagreement; tail accuracy stays 81.12%. Slippery early falls 18.00% -> 11.33%, high_grav 44.67% -> 34.67%; heavy rises 93.33% -> 97.33%. On the nine failures, prediction RMS averages 0.3280 versus oracle mode gap 0.1118; matched frozen windows have RMS 0.0951. Instantaneous margins are also negative, and no path has clipped input velocities. This motivates mean/support diagnosis, not merely changing posterior history or enlarging uncertainty.

**Decision / Next:** HOLD; retain v78 and preserve the conditional frozen-policy benefit without adopting disagreement. Use the nine server-side failed windows and matched frozen windows to diagnose 64-neighbor radii/composition, prediction-error directions relative to mode gaps, and fixed-pair energy/volume terms. Separate reference extrapolation from local-linear approximation error before a mean/support repair. No new simulation, seed expansion, coefficient tuning, actor integration or task dispatched here.

## Limitations
Three new environment seeds remain conditional on one source training seed, known mild modes and inherited calibration; this is not independent training replication or closed-loop return evidence. Accuracy equally averages paths/windows; mean risk is not a false-positive rate. Revised methods informed by these outcomes must treat v95 as development and use another untouched confirmation population. The old v61/v62/v76 counterexamples remain preserved separately.

Results: `results/general_ood_v95_fresh_seed/seed135190/fresh_seed.json`, `seed135191/fresh_seed.json`, `seed135192/fresh_seed.json`.
