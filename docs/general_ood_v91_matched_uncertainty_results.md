# v91 Matched uncertainty result

Formal task `t139108` completed on node007 CPU, exit 0, in 284.63s (4m 45s). All six folds / 3,072 inputs / 15,360 true-mode labels are present. Five frozen controls match v89 exactly, including all 30 fold comparisons. Sixty inner exclusions, 48 direct baseline queries and 2,400 physical Gaussian checks passed; the maximum physical-score discrepancy is 1.46e-11. Only the 223,285-byte scalar JSON returned; full error arrays stay remote.

| Reference condition | Accuracy | Classification NLL |
| --- | ---: | ---: |
| Prefix mean, frozen errors | 86.7969% | 1.1487982 |
| Prefix mean, matched errors | 87.1289% | 1.7821755 |
| Suffix mean, frozen errors | 86.1849% | 1.2371267 |
| Suffix mean, matched errors | 86.1198% | 1.2767342 |

Prefix loss worsens **55.13%** despite a 0.3320 percentage-point accuracy increase. Loss worsens in 5/6 folds and 24/30 fold-mode cells; cell accuracy improves in eight, regresses in 16 and is unchanged in six. All five mode-average losses worsen. Slippery accuracy increases 83.53% -> 85.42%, but its NLL worsens 1.4910 -> 2.2625; heavy accuracy regresses 94.50% -> 93.03%.

Seed135170 contributes 96.91% of the aggregate prefix loss increase. Its fold NLL rises 2.1787 -> 5.8616. For normal, covariance trace shrinks 3.4754 -> 0.1832 (5.27% of the old trace), log determinant changes by -10.1662, NLL rises 2.0830 -> 11.1185 and accuracy falls 69.14% -> 67.77%. Other modes' trace ratios on this seed are 5.04--11.15%. Added rows account for 14.36% of the prefix mean's neighbor selections, whereas the covariance tree contains no added rows.

Suffix loss worsens 3.20%; four of six fold losses and 20/30 fold-mode losses regress. Its aggregate accuracy decreases 0.0651 percentage points. Thus recomputing errors on the original support is not an adequate repair for either addition.

**Decision:** HOLD; retain v78 as the working baseline and the v89/v90 additions as development candidates. Neither matched-error condition is admitted to another trajectory replay. The calculations and exclusions are correct; the reference diagnostic rejects this particular uncertainty intervention.

**Next:** a reference-only support-matching pilot: include the same already-cached added rows in the error library and its nearest-neighbor lookup, while keeping both means, the original raw-covariance component and v83 mixture weight fixed. Apply whole outer/inner seed exclusion to all added labels. Compare against both frozen-error and original-support matched-error controls. This tests whether covariance support fails to transport uncertainty to the expanded mean's coverage; no follow-up task is dispatched here.

## Limitations
These reference folds and the frozen globally selected weight are development evidence, not independent confirmation or RL returns. Smaller inner fitting populations can affect error transport. The trace/coverage observations motivate a support-mismatch hypothesis but do not establish it as the sole cause; covariance orientation also matters. No evaluation trajectories, actor updates or manuscript tables changed.

Result: `results/general_ood_v91_matched_uncertainty/pilot/matched_uncertainty.json`.
