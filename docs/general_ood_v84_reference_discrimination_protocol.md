# v84 Reference discrimination diagnosis

Reuse the six v83 nested whole-seed fold caches, each with 512 input rows and five counterfactual next-state labels. Reconstruct each held-out observation's offset to every candidate mean using the frozen global forward coefficients and cached own-mode error. No inner fit, covariance fit, weight selection or evaluation-path loading is performed. Preserve raw weight 1, pure cross-seed weight 0 and the v83-selected mixture.

For all 25 true/candidate mode pairs, compute Gaussian log densities using the candidate's covariance and determinant. The shared physical Gaussian constant cancels in relative likelihoods. Report equal-seed/equal-mode accuracy, uniform-prior classification NLL, signed margins, confusion fractions and pairwise energy/volume decomposition. Focus on slippery versus compound/normal, heavy versus high_grav and high_grav versus normal; retain all five modes in the denominator.

Local tests cover cross-mode target/mean correspondence, candidate-specific Gaussian units, relative-score invariance and the distinction between true-density fit and classification. In the server task, verify reconstructed means against direct held-out local-linear predictions on eight rows per fold and reproduce all v83 fold/aggregate true-density losses. A mismatch stops interpretation and requires correcting reconstruction.

If reference holdouts exhibit the same ranking failures as v83 query replays, test a reference-only discrimination-aware selection objective next. If holdouts distinguish the modes but query failures persist, target reference-to-query geometry/coverage instead. No new seeds, temporal-rule changes, actor updates or paper tables. Only scalar JSON/logs return locally; source banks and fold arrays stay on node007, where the existing native runtime resides.

## Limitations
These are post-selection diagnostics on the same weight-selection folds, not independent confirmation. Outer labels are excluded from mean/covariance fits but contributed to the shared mixture weight. Uniform-prior instantaneous classification is not recursive query-path performance or RL return evidence.
