# v43 Normal Reference Results

2026-10-06: t135906 completed on node007 GPU0, exit 0, 279.19 seconds. Twenty-four independent normal references plus 200 paired tests: 380928 interactions, zero RL updates/new training seeds. Eleven focused tests passed. Config/runtime and test reward/termination/truncation traces match v42 exactly. Only 3983853 bytes of JSON synced; checkpoints stayed remote. Peak VRAM 232 MB, RAM 749 MB.

Broader normal references remove the catastrophic fit/calibration error gap: quadratic state-action fitting/calibration MSE in each model's own fitting-target units is 0.00774/0.00957 for fixed-minus2 and 0.05581/0.03709 for adaptive. These scales differ from v42; absolute MSEs across the two studies are not directly comparable.

Quadratic state-action plus episode-centered calibration detects every changed case within the first16 transitions: 20/20 seeds in each of four modes under both policies, with no physical endings there. No-change early alarms occur in 0/20 fixed-policy cases and 3/20 adaptive cases. Adaptive per-transition early exceedance fractions are 98.44% heavy, 86.25% slippery, 55.31% high_grav, 79.06% compound versus 1.25% no-change. Fixed-policy fractions are 100% for every mode versus 0% no-change.

Adaptive no-change exceedance fractions are 5.23% in validation256, 4.98% in first_post256, 3.83% in second_post256. Global calibration still mixes regimes; centering alone is insufficient. Linear episode calibration has 25.62% adaptive early no-change exceedance, while the state-only quadratic control detects only 13.44% of adaptive high_grav early transitions. All capacities/calibrations remain in the summary.

Judgment: useful development evidence for the observation mechanism; proceed to frozen belief integration, not a policy-improvement or submission claim.

Next: replay a fixed log-scaled calibration-normalized residual through the run-length filter and compare its current fixed half-normal likelihood with an observation model that can learn a persistent new residual level. Inspect switch response, sham resets, stabilization and implied beta before a matched-start/matched-budget RL feedback experiment. Large residual scores must not be treated as repeated changes: the present likelihood has a fixed zero center and does not infer a new stationary mean.

Limitations: one training seed and reused development tests; fitting/calibration coverage change together. Episode age is a coarse proxy, not latent-mode identification. Fixed-policy episode calibration still has 9.80% validation exceedance and is not uniformly calibrated. Deterministic action/state coupling prevents a causal action-identification claim. This is not an extreme50x control evaluation.

Summary: `results/general_ood_v43_normal_reference/summary.json`.
