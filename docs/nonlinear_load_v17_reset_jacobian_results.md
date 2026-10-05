# Nonlinear load v17: sampling helps initial regulation, not the full OOD gate

All four formal tasks t125062--t125065 exited 0, with no retry descendants. Runtime was approximately 34--51 minutes. Existing local JSONs total 6,290,517 bytes; no manual remote pull or checkpoint download was needed. The frozen summary validated all budgets, 9,600 controller episodes, paired paths, cost decompositions, identical within-seed fitted gains and common known-control returns.

## Results

Total-cost changes relative to the same-seed replay arm; negative is better. Each entry uses 100 paired paths, without pooling training seeds.

| Reset radius | Load severity | Seed11 | Seed12 |
| --- | --- | --- | --- |
| 0.2 | 1 | -0.090% | +0.025% |
| 0.2 | 50 | +0.045% | +2.564% |
| 0.5 | 1 | -1.056% | -0.216% |
| 0.5 | 50 | -0.455% | +1.830% |
| 1.0 | 1 | -6.089% | -1.281% |
| 1.0 | 50 | -4.546% | +0.130% |

**HOLD.** Radius1 repair failed at seed12/severity50, despite lower early state cost in all four radius1 blocks. Small-radius retention failed in three of four blocks. Own-fitted added value passed at radii0.2/0.5 but failed in all four radius1 blocks (actor excess costs: seed11 +0.050005/+0.001189; seed12 +0.029381/+0.031249 at loads1/50). Do not proceed to learner-comparator replication or retune this cohort. Historical v13/v15 HOLD remains unchanged.

## Mechanism And Next Step

At radius1, initial physical-input action-Jacobian squared error against own fitted feedback fell about77--80% in every seed/load block. Early state-cost changes were seed11 -0.483495/-0.513773 and seed12 -0.116406/-0.116664 at loads1/50. Thus the intervention addressed initial reference alignment and state regulation. For seed12/severity50, however, late state cost rose about0.0729 at every radius, plus late command cost about0.0075--0.0078. At radius0.2, 97/100 paired paths worsened: this failure is not confined to a few largest outliers.

The stored fitted load-gain blocks are seed11 [[0.299697,0.165098],[0.139967,0.275769]] and seed12 [[0.140621,0.169330],[0.095890,0.080867]], although the environment applies loads and actuator forces componentwise. At radius0.2/severity50 the fitted controller costs3.044666/3.747248 for seeds11/12, versus common known local/inverse costs2.704497/2.562471. These observations make reference identification the next bottleneck to test. The existing degree-one fit omits the actuator's cubic action term; cross-axis fitted load gains and weak seed12 compensation are consistent with misspecification or poor identification, but do not yet establish its cause.

Next, run a read-only identification diagnostic: reproduce the original seed11/12 fits and verify their gains, then compare a cubic-actuator-aware fit on the same collected transitions, with independent physical-dynamics validation and direct-feedback evaluation at loads1/50. Keep training checkpoints on servers and return small JSONs. No actor training, sampling/weight sweep or seed expansion before resolving reference quality.

## Limitations

Two training seeds and reused evaluation paths do not establish general transfer or learner-baseline superiority. Trajectory alignment summaries use each actor's own visited states, so their differences are descriptive, not state-matched derivative comparisons. Known controllers are feasible references, not certified optima. Sampling and reference-quality hypotheses remain separate; improved initial alignment does not demonstrate that sampling caused the reference deficiencies.

Evidence: `results/nonlinear_load_v17_reset_jacobian/{task_observations,summary}.json` and the four per-arm/per-seed evaluation JSONs. Reproduce the frozen summary with `python3 scripts/summarize_nonlinear_load_v17.py`.
