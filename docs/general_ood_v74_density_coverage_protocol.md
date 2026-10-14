# v74: density-only slippery-state coverage

Keep the v72 forward coefficients/scales and v73 window16/recursive rules
(hazard 0.02). Retain the original bank as a baseline. Compare normal-state
versus slippery-state density extensions with the same 4,400 labels/mode,
appended to the same 18,600 base labels/mode. Keep 64 neighbors, local OAS,
ridge and physical Gaussian normalization. No forward fitting or actor feedback.

Density seeds 135150/135151: paired normal/slippery paths, 6,000 steps, switch
3,800; use only their 2,200-step suffixes with same-state/action all-mode
counterfactual targets. Independently recalibrate both readouts for each new
bank on the original 12,000 normal-null transitions/rules. Original curves
remain unchanged for the baseline. Freeze all references before new test
seeds 135160/135161, each with all five modes and the same time contract.

Budget: 24,000 reference + 60,000 test actual transitions; 420,000 shadows;
504,000 simulator calls, zero optimizer updates. Retain exact true-mode replay,
paired prefixes and the existing normal-risk/positive-risk-and-mode-margin
screen. Report early/tail accuracies and conditional mean error separately;
tail regressions and updating-policy failures remain visible. Prior evaluation
traces are development only. Raw arrays/checkpoints stay remote; scalar JSON
and logs only. CPU, eligible Linux input nodes, no node/GPU preference/Windows.

## Limitations

One source policy, mild known modes and privileged labels. Density coverage
changes local means and covariances together; this is not control confirmation.
