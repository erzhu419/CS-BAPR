# Nonlinear load v19: reference repair improves OOD cost, not actor added value

All four tasks t126897--t126900 exited0 on automatically selected Linux GPUs, in60--82 minutes. All evaluation JSONs were already synced locally (6,336,453 bytes); no manual remote pull or checkpoint download was needed. The frozen summary validated400,000 learner interactions,396,004 updates,40,000 identification and4,000 validation interactions,9,600 controller episodes, matched identification and paired known controls.

## Results

Actor total-cost changes versus the same-seed linear-reference arm; negative is better. Each block uses100 paired paths.

| Reset Radius | Load Severity | Seed13 | Seed14 |
| --- | --- | --- | --- |
| 0.2 | 1 | -0.903% | -0.318% |
| 0.2 | 50 | -33.962% | -46.907% |
| 0.5 | 1 | -1.390% | -0.787% |
| 0.5 | 50 | -26.879% | -38.747% |
| 1.0 | 1 | -1.186% | -1.300% |
| 1.0 | 50 | -15.627% | -24.494% |

**HOLD.** Total cost decreased in all12 seed/radius/load blocks, with92--98 of100 paired OOD paths improving in each block. Small-radius retention passed. Radius1 repair failed only at seed13/load1 early state cost (+0.007827), despite lower total cost. Own-reference added value failed in all12 blocks: actor excess cost was0.033--0.398% at radius0.2,0.403--0.911% at radius0.5, and1.733--3.211% at radius1. Learner-comparator replication is not admitted.

## Interpretation And Next Step

Corrected identification removed most reference error: ID physical validation RMSE fell from0.0131949/0.0133883 to9.80e-8/1.90e-6 at seeds13/14. Initial actor output-force load-Jacobian cancellation residual fell in all12 blocks; at radius1/load50 it changed1.483715->0.095837 and2.581821->0.101612. Direct fitted-controller OOD costs also fell15.99--49.61%. Together, these results support reference misspecification as a material source of previous OOD failures; the actor's remaining deficit is not evidence of useful improvement over the corrected reference.

At radius1, actor excess total costs versus its own reference are0.163334/0.172781 for seed13 and0.119344/0.129397 for seed14 (loads1/50). Early state costs increase0.184--0.253, while command costs decrease0.057--0.089; late state-cost changes are only-0.001025--+0.000289. The remaining deficit is concentrated in initial physical-state regulation. The existing penalty constrains derivatives with respect to all six observations. Whether physical-state derivative matching restricts useful nonlinear regulation remains a hypothesis, not a demonstrated cause.

Proposed next experiment: keep the corrected cubic-local reference and environment fixed; compare full six-input Jacobian matching against matching only the two observed-load input columns. Keep the same summed squared-error definition and weight0.01 on retained columns, reset-balanced sampler, architecture, training budget and last-checkpoint selection. Use fresh seeds15/16, two arms per seed. Require radius1 total/early-state improvement, zero-margin radius0.2 retention and own-reference added value in every seed/load block before learner-comparator replication. This isolates the constraint direction rather than changing identification, environment and objective together. No v20 task has been submitted.

## Limitations

Two seeds, reused evaluation paths and one fully observed nonlinear-load environment do not establish general RL or BAPR/RE-SAC superiority. The direct controllers are feasible references, not certified optima. The cubic seed14 fit still prunes one positional load coefficient0.00125; do not pool it with the nearly exact seed13 fit. Trajectory alignment uses each actor's visited states. Historical v13/v15/v17 HOLD and manuscript tables remain unchanged.

Evidence: `results/nonlinear_load_v19_reference/{task_observations,summary}.json` and the four per-reference/per-seed evaluation JSONs. Reproduce with `python3 scripts/summarize_nonlinear_load_v19.py`.
