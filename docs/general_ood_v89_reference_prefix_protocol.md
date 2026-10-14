# v89 Matched reference mean coverage

Use only the two saved v76 updating-normal reference traces, seeds 135170/135171. `prefix_mean_mix` adds 1,100 evenly spaced rows from each normal prefix `0:3800`. `suffix_mean_mix` adds the 1,100 odd rows `3801:6000:2` from each same normal trace; the original bank used even suffix rows. Both additions contain 2,200 inputs and 11,000 cached five-mode counterfactual labels. Load no paired slippery prefix and no evaluation trace.

Extend only the mean-fitting bank, from 23,000 to 25,200 rows. Keep the v78 local-linear recipe (64 neighbors, ridge 0.064), v83 candidate-specific covariance/weight, forward coefficients and scaling fixed. Use the identical six outer-fold inputs/labels. Remove every row of each outer seed, including its additions, before mean fitting; added training counts must match between conditions.

Tests check index uniqueness/non-overlap, counterfactual ordering, unchanged base fit/scaling, full seed exclusion and frozen Gaussian covariance. Native source checks reconstruct the 2,200 existing normal-suffix bank rows and all five residual labels. Native smoke uses outer seed135170 to exercise exclusion of added rows, verifies eight direct baseline means and all three old control summaries, then scores both additions. A correspondence mismatch blocks formal dispatch.

Formal pilot compares NLL, accuracy, signed margins, confusion and normalized-target error RMS by fold/mode, retaining raw/pure-cross-seed/v83-mixture controls. Compare prefix both against the unchanged mixture and against the equal-count suffix addition before considering a query replay. Only scalar JSON/logs return locally; arrays stay on node007. There are no new environment steps, counterfactual simulations, actor updates, parameter tuning or manuscript changes.

## Limitations
These are previously inspected reference folds, not independent confirmation or RL returns. Both additions increase the used reference-label budget. Prefix versus suffix changes the sampled state/action distribution, so a contrast is not a pure clock-time causal effect. Covariance support remains unchanged to isolate the mean intervention.
