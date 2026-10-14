# v68: known-mode transition likelihood screen

Frozen before dispatch on 2026-10-08. Keep v66 unchanged as the diagnostic
control. v67 is not promoted. No actor/belief feedback or manuscript changes.

## Reference And Observer

- Source: v40 fixed-minus2 Hopper training seed 1, 99,745 updates;
  v52 shared context (100,000 transitions) and fit-only replay (6,144 entries).
- Freeze the stochastic policy. Use the native 1,000-step TimeLimit and
  existing normal/heavy/slippery/high_grav/compound profiles, constant from
  the start of each reference trajectory.
- Seed 135000: 6,000 fitting transitions per mode (30,000 total).
- Seed 135010: 6,000 independent residual-density calibration transitions
  per mode (30,000 total).
- Seed 135020: 6,000 normal-only transitions for departure threshold
  calibration, separate from both model fitting and density calibration.
- Total new environment interactions: 66,000. Actor/critic updates: zero.
  Online context observes transitions; RL replay is not augmented.
- Fit five quadratic state/executed-action ridge delta-state predictors.
  Use pooled fitting-only input/target scales and fixed mean-loss ridge 0.001.
  Fit each residual mean and OAS covariance on its density-calibration split;
  add the existing relative covariance ridge 0.001. Include the full Gaussian
  log determinant and common physical-coordinate Jacobian in every density.
- Use the causal last-16 mean log densities, including the current observed
  transition. Report uniform-prior softmax mode weights as tempered diagnostic
  weights, not accumulated Bayesian posterior probabilities.
- Departure signal: softplus of the equal-mixture changed-mode versus normal
  log likelihood ratio. Calibrate initial/later-episode age thresholds using
  only the independent normal-null split and existing 95%/16-step settings.
  Histories persist across episode boundaries. No mode/change label enters
  the observer.

## Replay And Decision

Replay the existing 12 v62 test cells and 5 v61 development cells separately.
Read only completed server-side traces. Retain v66's original readout summaries.
No threshold, profile or hyperparameter selection on these replay cells.

The unchanged departure gates require every frozen/updating normal first800,
full, last1000 and last64 mean <= 0.10, and every paired changed first_post50
and last64 contrast > 0. Report mode accuracy, confusion fractions and the
true-mode versus best-other mean log density in those same windows. A mode
identification screen also needs that true-mode margin positive in every
changed pair/window before considering sequential belief. HOLD otherwise.

CPU scheduler task, no Windows or placement preference. Reference/source data
currently reside on node007, so eligibility is limited to that data host.
Resume uses existing agent/RNG/context/environment checkpoint machinery.
Only scalar JSON/logs sync locally; traces and reference matrices stay remote.

## Limitations

This adds labelled physical-mode references and simulator-profile access.
Changed profiles may be OOD to the actor but are known to the observer; this
does not solve arbitrary unseen physical shifts. References use a frozen
policy, whereas replay also includes updating policies. Evaluation traces
have already been inspected and come from one source training seed. Any
passing result is a development screen, not independent or RL-return evidence.
