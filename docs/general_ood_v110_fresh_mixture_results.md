# v110: confirmation gain is small; normal regression remains

Tasks `t141412`/`t141413`/`t141414` completed in 35m13s / 36m07s / 35m13s,
exit 0, with all scalar results synced (2,063,507 bytes). All 30 paths and
48 changed windows are present. References match frozen v109 metadata; budgets
are 180,000 real transitions, 900,000 shadow calls and 90,000 optimizer updates.

Primary condition: `suffix_disagreement_mix`; equal early/tail window weights.

| Metric | v98 | v106 | Normal mixture |
| --- | ---: | ---: | ---: |
| Updating recognition | 82.01% | 82.77% | 82.71% |
| Updating early recognition | 66.50% | 68.67% | 68.67% |
| Updating tail recognition | 97.53% | 96.88% | 96.74% |
| Nonpositive updating margins / 24 | 4 | 4 | 4 |
| Maximum updating normal-window mean risk | 0.00433 | 0.00647 | 0.00759 |
| Frozen recognition | 98.54% | 98.52% | 98.52% |

Mixture updating gains by seed are +1.859 / -0.031 / +0.250 points. The mean
gain is +0.693 points; its paired environment-seed t interval is [-1.841, 3.227]
points (n=3, df=2). These are not independent source-training repetitions.

Seed135252 normal full-history risk rises 0.00313 to 0.00759. In its unchanged
normal first_post50 window, accuracy drops 86% to 38%, mostly false heavy,
and mean risk rises 0.07965 to 0.62293. Episode3 risk rises 0.00739 to 0.03553.
Its slippery/high_grav/compound early accuracies remain 2% / 0% / 0%; high_grav
is predicted heavy, whereas slippery/compound are mostly predicted normal.
Support-only and trace-matched controls retain small aggregate recognition
gains (+1.04 / +1.25 updating points) but also show normal-risk regressions.

**Decision:** confirmation does not justify adoption or actor integration.
Retain v98. Next diagnose the four failed early windows with frozen-policy
pairs and six normal controls: compare model errors with simulator mode gaps
and correct-center density attribution under the unchanged covariance.
No seed expansion, coefficient search or new reference fitting.

**Limitations:** one source policy and known mild Hopper modes; reference
budgets differ. Risk means are not false-positive rates. Correct simulator
centers are a diagnostic intervention, not a deployable estimator or a return test.

Artifact: `results/general_ood_v110_fresh_mixture/summary.json`.
