# v93 Mean disagreement result

Formal task `t139181` completed on node007 CPU, exit 0, in 158.96s (2m 39s). All six folds / 3,072 inputs / 15,360 true-mode labels per condition are present. Nine controls match v92, including 54 fold comparisons. Sixty member exclusions, 120 direct member predictions and 2,400 physical Gaussian checks passed. Only the 326,451-byte scalar JSON returned; member arrays remain remote.

| Reference condition | Accuracy | Classification NLL |
| --- | ---: | ---: |
| Prefix mean, frozen errors | 86.7969% | 1.1487982 |
| Prefix mean, v92 error support | 87.4023% | 1.7481169 |
| Prefix mean, v93 disagreement | 87.2721% | 1.5314313 |
| Suffix mean, frozen errors | 86.1849% | 1.2371267 |
| Suffix mean, v92 error support | 86.0482% | 1.2769449 |
| Suffix mean, v93 disagreement | 85.8203% | 0.9981644 |

Suffix NLL improves **21.83% over v92 and 19.32% over frozen errors**. Five of six fold losses improve against both controls; 23/30 fold-mode losses improve over v92 and 18/30 over frozen errors. All five mode-average losses improve. Heavy accuracy rises 93.95% -> 94.60% against frozen errors. Overall accuracy decreases 0.3646 percentage points; slippery and compound each decrease 0.8789 points. Seed135171 compound is an adverse cell: NLL increases 0.2341 and accuracy decreases 3.5156 points. Lower aggregate log loss is not uniform dominance.

Prefix NLL improves 12.40% over v92 but remains **33.31% worse than frozen errors**. Five folds improve against frozen errors, but seed135170 fold NLL remains 5.4199 versus 2.1787. Its normal NLL is 10.1783 versus 2.0830 frozen; covariance trace increases 0.1890 -> 0.2936, still only 8.45% of the frozen trace. All five mode-average losses remain worse than frozen errors.

**Decision:** HOLD, retain v78; admit only `suffix_disagreement_mix` to a matched development replay. Do not adopt the readout, advance the prefix uncertainty condition or tune the disagreement coefficient.

**Next:** use the same 65 saved paths and 12,000 normal-null transitions, preserving temporal rules. Fit full-bank cross-seed errors using only the 25,200 suffix reference rows; deployment has six source-deletion members rather than the five remaining after outer exclusion here. Retain v92 without disagreement and reused v78/v83/v90 controls. Add a trace-matched control `(1 + trace(D)/trace(Sigma_v92))*Sigma_v92` alongside `Sigma_v92 + D`; both share the same computed disagreement and trace, but only the latter adds its directional shape. Recalibrate each new readout on the same normal-null population. Inspect recursive negative windows, early/tail accuracy, normal risk and slippery/compound regressions before adoption. No task is dispatched in this assessment.

## Limitations
These are reused development folds with a globally selected frozen weight, not independent confirmation or RL return evidence. Unequal seed blocks make deletion sensitivity an empirical second moment, not a calibrated variance. Improved log loss may reflect broader covariance rather than a specifically useful directional component; the matched rescaling control tests that distinction. No actor, environment or manuscript changes.
