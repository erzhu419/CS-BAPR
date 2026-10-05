# Nonlinear load v9: completed learner comparison

All eight tasks `t120342`--`t120349` completed with exit 0 on Linux CUDA workers, taking 31.4--50.0 minutes each. Final JSONs match frozen configurations, 500 episodes, 100,000 learner interactions, 99,001 updates and paths 50000--50099. There are 15,600 evaluation episodes including direct controls; all eight small JSONs were already local (967,561 bytes). This completion pass transferred no remote data or checkpoints.

Mean total cost, lower is better. Learner/fitted means average training seeds 5/6; each seed uses 100 paired paths per severity. Known controls use the same paths and match the frozen pilot.

| Method | 1x | 2x | 5x | 10x | 20x | 50x |
|---|---:|---:|---:|---:|---:|---:|
| Centered/no-NAU + symbolic | 0.258902 | 0.260641 | 0.276662 | 0.346539 | 0.716351 | 3.791397 |
| Centered/no-NAU, no symbolic | 41.779312 | 41.759000 | 42.004077 | 42.762762 | 44.788815 | 71.389913 |
| BAPR | 21.447090 | 21.589526 | 22.854877 | 27.493289 | 48.118692 | 388.802729 |
| RE-SAC | 40.536156 | 40.576268 | 41.207663 | 46.090702 | 111.906537 | 738.912500 |
| Candidate's own fitted feedback | 0.260825 | 0.262602 | 0.278758 | 0.349439 | 0.723162 | 3.779701 |
| Known local-linear feedback | 0.255739 | 0.257130 | 0.269106 | 0.322244 | 0.597803 | 2.859103 |
| Known inverse-actuator feedback | 0.275984 | 0.277383 | 0.289153 | 0.341993 | 0.611196 | 2.726351 |

## Decision

The frozen learner-comparison gate passes against all three learners, with 2/2 matched-seed wins at ID/50x and every intermediate severity. At 50x the candidate reduces cost by 94.69% versus no-symbolic, 99.02% versus BAPR and 99.49% versus RE-SAC. Its seed costs are 4.163617/3.419177 (sample SD 0.526398); no-symbolic costs are 98.509312/44.270514. Retain the centered/no-NAU symbolic candidate here; the centering-only alternative is not supported by this round.

The separate added-RL-value gate fails. ID learner-minus-own-fitted mean returns are +0.002292/+0.001553, but 50x gaps are -0.040911/+0.017518. Mean 50x learner cost is 0.31% higher than its fitted controls. Seed5 wins 70/100 paths and seed6 81/100, yet seed5's larger negative gaps outweigh its frequent small improvements; paths 50002 and 50044 alone have gaps -1.697699/-1.535003. Path wins do not replace the frozen mean-return gate.

Candidate reset-action RMSE against its fitted reference is 0.009520 at ID and 0.009889 at 50x. Together with the cost gaps, this is consistent with closely recovering the biased reference rather than substantially correcting it. Candidate 50x cost exceeds known local-linear/inverse controls by 32.6%/39.1%, respectively; this remaining performance gap is not a lack of feasible control authority.

## Next Step

Keep this environment, weights and failed gate fixed; do not expand seeds or increase severity. Diagnose the existing seed5/6 checkpoints on their Linux workers, without pulling checkpoints: compare actor, own fitted, known local-linear and inverse controls on the same paths, decomposing physical-state versus command-effort cost and inspecting current-load feedback/Jacobian errors along trajectories. Include the listed negative-gap paths and aggregate all 100 paths rather than selecting only favorable cases. Return only small aggregates. Use that diagnosis to decide whether a critic-only versus Jacobian-only symbolic ablation is warranted; no new training was submitted by this completion pass.

This diagnosis is now completed in `docs/nonlinear_load_v9_diagnostic_results.md`: the remaining 50x gap is dominated by state cost, and both actors retain their fitted reference's biased load-feedback derivatives. The next justified training test is the missing critic-only/Jacobian-only ablation cells, not seed expansion. The read-only diagnostic submitted no training.

## Limitations

Two training seeds do not establish statistical significance; 100 evaluation paths are not 100 independent trained policies. This is a synthetic nonlinear-actuator task with observed load and known equilibrium, not standard-benchmark or bus-domain transfer. Symbolic tasks use 11,000 extra identification/validation interactions and combine critic/Jacobian constraints, whose contributions are not separated. Known controls are privileged feasible references, not optima. GPU models differ (2080 Ti/3080 Ti), while pinned package versions match. Reset-state alignment does not describe full trajectories; it also uses different references across symbolic/non-symbolic arms. V7/v8 results, all failed gates and manuscript tables remain unchanged.

Reproduce with `python3 scripts/summarize_nonlinear_load_v9.py`. Evidence is in `results/nonlinear_load_v9_training/summary.json`, `seed_mean_returns.csv` (all 48 seed/severity means), `task_observations.json`, `completion.json` and the eight final evaluation JSONs.
