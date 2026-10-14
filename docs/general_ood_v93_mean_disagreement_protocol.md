# v93 Source-seed mean disagreement

v92 expanded error support leaves prefix classification NLL 52.17% worse than frozen errors; seed135170 normal covariance trace remains only 5.44% of its frozen counterpart. Test whether source-seed mean sensitivity supplies useful missing prediction uncertainty.

At each query, first remove the entire outer seed. Retain the expanded v89 mean `mu`, original raw covariance, cached v92 errors, 64-neighbor rules, ridge alpha 0.064 and v83 raw weight 0.24619454443573643. Predict `mu_minus_g` after removing each of the five remaining source seeds, including all its original and added rows. Add `D = mean_g[(mu_minus_g - mu)(mu_minus_g - mu)^T]` to the unchanged v92 covariance, with coefficient exactly one. Do not replace `mu` with the member average; the second moment includes both centered member spread and the squared member-average displacement. All terms use normalized target coordinates; physical Gaussian scores include log determinant and scale conversion.

Two conditions: `prefix_disagreement_mix` and `suffix_disagreement_mix`. Reproduce all nine v92 controls. Formal evaluation uses the same six outer folds, 512 fixed inputs and five true-mode labels per fold. Compare each new condition with its v92 and frozen-error counterparts by fold/mode classification NLL and accuracy. Record base/added/final trace, centered/bias decomposition and log-determinant changes, particularly seed135170 normal and aggregate heavy.

The one-fold native smoke uses seed135170, eight new-condition queries and all 512 control queries. Require nine reproduced controls, whole-source-seed exclusion, direct member correspondence, unchanged means, the additive formula and physical Gaussian correspondence before formal dispatch. Performance is assessed only on the formal pilot; no strength tuning, new data or replay in this step.

Stage a fresh frozen runtime. Run CPU-only on node007 because native caches reside there. Member arrays stay in excluded server-side `run_checkpoints`; return scalar JSON/logs only. Keep HOLD and existing actor/environment/manuscript state until formal assessment.

## Limitations
These reused folds and the globally selected weight are development evidence. Seed blocks are unequal, so deletion sensitivity is not a calibrated predictive variance or an unbiased jackknife estimate. The added component can double-count prediction error or impair discrimination. No independent confirmation, recursive adaptation or RL-return claim follows from this pilot.
