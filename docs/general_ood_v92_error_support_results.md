# v92 Error support result

Formal task `t139152` completed on node007 CPU, exit 0, in 210.45s (3m 30s). All six folds / 3,072 inputs / 15,360 true-mode labels per condition are present. Seven controls match v91, including 42 fold comparisons. Eighty seed-exclusion checks, 80 direct error-row checks and 2,400 physical Gaussian checks passed. Only the 293,367-byte scalar JSON returned; error arrays remain remote.

| Reference condition | Accuracy | Classification NLL |
| --- | ---: | ---: |
| Prefix mean, frozen errors | 86.7969% | 1.1487982 |
| Prefix mean, v91 original error support | 87.1289% | 1.7821755 |
| Prefix mean, v92 expanded error support | 87.4023% | 1.7481169 |
| Suffix mean, frozen errors | 86.1849% | 1.2371267 |
| Suffix mean, v91 original error support | 86.1198% | 1.2767342 |
| Suffix mean, v92 expanded error support | 86.0482% | 1.2769449 |

Expanded prefix error support improves NLL 1.91% over v91, but remains **52.17% worse** than frozen errors. Relative to frozen errors, four of six fold losses and 20/30 fold-mode losses worsen; all five mode-average losses worsen. Accuracy rises 0.6055 percentage points overall, but heavy falls 94.50% -> 92.94%. Suffix NLL worsens 3.22% against frozen errors and 0.0165% against v91, with lower accuracy.

Seed135170 accounts for 97.67% of the net prefix loss increase against frozen errors. Its fold NLL remains 5.6909 versus 2.1787 frozen. Normal NLL is 10.9698 versus 2.0830 frozen and 11.1185 in v91. Normal covariance trace changes only 0.1832 -> 0.1890, still 5.44% of the frozen trace; added rows enter 14.36% of its error-neighbor selections. Including additions does not repair the dominant uncertainty failure.

**Decision:** HOLD; retain v78. Stop this error-support expansion branch and do not admit either condition to trajectory replay. Means, raw geometry and weight correspond correctly; the tested covariance intervention fails on the development references.

**Next:** test source-seed mean disagreement rather than further support expansion or weight tuning. With the outer seed absent, predict each fixed input after removing each remaining source seed in turn. Keep the outer mean unchanged and add the equal-seed second moment of these mean displacements to the v92 covariance. Compare prefix/suffix against their frozen-error and v92 controls on the same six folds; geometry must use query inputs only, without true labels. This tests missing mean-estimation uncertainty, not a mean correction. No follow-up task is dispatched here.

## Limitations
Reused development folds and a globally selected frozen weight are not independent confirmation or RL return evidence. Covariance traces alone do not establish directional calibration. A mean-disagreement second moment measures sensitivity to unequal source-seed populations; it is not an established calibrated variance, and adding it may double-count error or worsen discrimination. No actor, environment or manuscript changes.
