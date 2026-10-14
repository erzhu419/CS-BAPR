# v87 Shared covariance result and next step

Task `t138443` completed on node007 CPU, exit 0, in 85.43 seconds. Four local tests passed. All 18 fold/control comparisons reproduce v84; candidate determinant differences are exactly zero. The 100,287-byte scalar result covers six folds, 3,072 inputs and 15,360 true-mode labels. No traces, training or actor changes were included.

Against v83's selected mixture, shared covariance reduces accuracy from 86.0677% to 84.1146% and increases classification NLL from 1.2285603 to 1.3151732 (+7.05%). Accuracy falls in all six folds and all five mode averages. Fold NLL improves in three folds and regresses in three; 14/30 fold-mode losses improve and 16 regress.

Normal, heavy and high-gravity average NLL improve, but slippery rises from 1.5250 to 2.5600 and compound from 1.1168 to 2.2271. Compound accuracy falls from 87.4023% to 81.3151%. Two earlier localized NLL regressions are repaired versus raw (135171 normal and 135092 heavy); both slippery cases remain worse than raw. Reduced slippery-to-compound confusion therefore does not establish a better calibrated classifier.

**Decision:** HOLD, retain v78 as the working baseline. Do not advance this shared-covariance condition to the 65-path replay or expand seeds. Candidate-specific covariance is not merely a nuisance determinant term; tying it also loses useful discrimination.

**Next:** test a reference-only cross-seed residual mean-bias correction, keeping v83's candidate-specific covariances and weight frozen. Exclude the outer seed from every inner mean/error fit used by the correction. Compare unchanged and corrected means by fold/mode NLL, signed margins and confusion, preserving normal and heavy controls before considering replay. No additional task has been submitted in this round.

## Limitations
These folds also supported earlier weight selection and diagnostics; this is development evidence, not independent confirmation. The pilot evaluates instantaneous reference classification, not recursive query behavior or RL returns. The negative result rejects this particular shared-covariance construction, not every covariance-sharing method.
