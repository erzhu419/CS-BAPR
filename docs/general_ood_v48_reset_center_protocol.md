# v48: normal-only startup correction and first50 response

Keep v47 policies, references, physical profiles, 1024/1280 switch timings, 1792
budget and TimeLimit1536. Compare global-robust against reset-centered-robust.

Fit one residual offset from normal calibration transitions at observable age0:
normal startup residual mean minus global residual center. Subtract it only at
age0 before the unchanged global scale/q95 and log1p transform. Do not discard
observations, clear history, reset belief, change hazard/tails, or use test modes.

Retain same-past original/corrected likelihood counterfactuals at every startup.
Add full1792 normal controls to the six v47 branches and save 50 individual
changed-minus-stay penalty values after the second switch. Retain majority events,
soft window/penalty statistics and physical endings. Old global control must
exactly reproduce v47; normal overlap must reproduce v46.

Preflight: 2 evaluation seeds per policy, 74752 interactions. Formal: existing
20 evaluation seeds per policy, 575488 interactions including 24 reference and
280 test rollouts. Two likelihood arms share each physical trajectory.

Only scalar JSON/logs return; checkpoints stay on node007, idle GPU selected by
scheduler without preference. No Windows node, new training seed, RL update,
environment tuning or paper changes. Failed startup correction or slow response
will remain negative results; formal parameters are not tuned on preflight.
