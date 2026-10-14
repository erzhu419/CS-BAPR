# v109: normal-only density mixture is the confirmation candidate

Task `t141297` completed in 7m45s: 95 paths / 570,000 cached transitions,
855 control path-condition comparisons and nine calibration comparisons passed.

Primary condition: `suffix_disagreement_mix`; equal early/tail window weights.

| Metric | v98 | v106 | Hard hybrid | Normal mixture | All-mode mixture |
| --- | ---: | ---: | ---: | ---: | ---: |
| v95 updating recognition | 68.12% | 74.27% | 72.19% | 72.19% | 70.44% |
| All updating recognition | 81.39% | 84.69% | 83.87% | 83.88% | 83.11% |
| Six old failures: recognition | 14.44% | 30.55% | 22.89% | 23.89% | 18.03% |
| Maximum updating normal-window mean risk | 0.00438 | 0.03205 | 0.00942 | 0.00472 | 0.00446 |
| Maximum all-normal-window mean risk | 0.01002 | 0.03205 | 0.01086 | 0.00866 | 0.00760 |
| Historical slippery tail: recognition | 32.81% | 65.63% | 65.63% | 65.63% | 60.94% |

Normal-only mixing retains a 4.07-point v95 updating gain versus v98 and removes
most of v106's normal regression. Across the three covariance conditions, v95
updating gains are 4.53 / 4.07 / 4.68 points. In the primary condition, 21/69
normal windows have higher mean risk than v98; the largest increase is
0.000746 (seed135191, updating, last1000). The updating maximum rises 7.69%,
whereas the all-normal maximum falls 13.59%; these are different populations.
Mean full-normal accuracy rises 99.11% to 99.20%. All v95 frozen changed
accuracies remain 98.75%. Seed135192 updating recognition remains only 49.63%.

**Decision:** freeze the normal-only mixture for fresh environment-seed
confirmation against v98 and v106. Stop tuning on these 95 paths. Keep the
production readout unchanged pending confirmation and subsequent actor tests.

**Limitations:** development evidence, one source policy, unequal reference
budgets and reused normal calibration. Mixture members share data. Risk means
are not false-positive rates; no actor, return or cross-environment gain follows.

Artifact: `results/general_ood_v109_density_mixture/formal/density_mixture.json`.
