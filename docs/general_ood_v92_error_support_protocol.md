# v92 Error support matching

v91 retained expanded means but estimated their errors only on the original support; prefix classification NLL worsened 55.13%. This pilot tests whether including the same cached additions in the error library and its nearest-neighbor lookup repairs that mismatch.

Two conditions: `prefix_support_mix` and `suffix_support_mix`. Both keep the v89 25,200-row mean, the original 23,000-row raw-covariance support, 64 neighbors, ridge alpha 0.064 and v83 raw weight 0.24619454443573643. Only the error component gains 2,200 cached inputs / 11,000 labels. No new simulator data or optimization.

Hold out each entire outer seed before forming either bank. For each error row, predict from the expanded mean bank without any original or added rows of its source seed. Reuse the identical v91 original-row errors and calculate only added-row errors. Verify representative rows by direct source-seed-excluded predictions. Error covariance uses expanded-bank neighbors and OAS second moments (`assume_centered=True`); raw covariance continues to use original-bank neighbors. Means, forward models, scales and mixture weight are fixed.

Compare seven reproduced controls and two new conditions on the same six outer folds, 512 fixed inputs and five true-mode labels per fold. Report classification NLL, accuracy, fold/mode deltas, covariance traces and added-error neighbor fractions. The one-fold native smoke checks eight new-condition inputs, all 512 control inputs and physical Gaussian correspondence. Formal admission requires control reproduction, whole-seed exclusion and formula correspondence; smoke performance is not an admission criterion.

Run on node007 CPU because the native inputs and caches reside there. Only scalar JSON/logs return locally; arrays stay in excluded `run_checkpoints`. Freeze a new runtime directory before staging. No trajectory replay, actor feedback or manuscript changes in this step. Keep HOLD pending formal analysis; any replay admission needs lower loss across folds without worsening the known difficult modes.

## Limitations
These are reused development folds with a globally selected frozen weight, not independent confirmation. Cross-fitted errors and deployment means use different fitting populations. Reference classification alone does not establish recursive adaptation or RL returns.
