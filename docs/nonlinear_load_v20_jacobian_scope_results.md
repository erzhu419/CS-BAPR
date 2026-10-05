# Nonlinear load v20: load-only constraint fails

All four formal tasks t126954--t126957 exited0 in52--83 minutes. All evaluation JSONs were already local (6,311,851 bytes); no remote pull or checkpoint download was needed. The frozen summary validated400,000 learner interactions,396,004 updates,40,000 identification and4,000 validation interactions,9,600 controller episodes, matched identification and paired fitted/known controls.

## Results

Load-only actor total cost divided by the same-seed full-scope actor cost. Values above1 are worse; each block uses100 paired paths.

| Reset Radius | Load Severity | Seed15 | Seed16 |
| --- | --- | --- | --- |
| 0.2 | 1 | 3.842 | 25.900 |
| 0.2 | 50 | 1.239 | 3.166 |
| 0.5 | 1 | 5.553 | 11.536 |
| 0.5 | 50 | 2.695 | 4.793 |
| 1.0 | 1 | 4.907 | 5.973 |
| 1.0 | 50 | 3.824 | 4.410 |

**HOLD.** All12 total-cost blocks worsened; all three gates failed. In each radius1 block,100/100 paired paths worsened. ID paths worsened100/100 at every radius; OOD counts were98/99/100 for seed15 and100/100/100 for seed16 at radii0.2/0.5/1. Load-only failed own-reference added value in every block. Do not expand seeds or learner comparisons for this intervention.

## Interpretation

At radius1, load-only adds7.02--7.30 early state cost and13.52--18.60 late state cost. Command-cost savings cannot offset the state-regulation deficit. Initial physical-input action-Jacobian squared error rises from0.0205 to3.5160 for seed15/ID and0.0322 to7.0537 for seed16/ID. This supports the full constraint's important regulation role and rejects physical-column removal as the proposed repair.

The full-scope actors remain close to their own fitted references: radius1 excess cost is1.096%/0.561% for seed15 and1.801%/1.211% for seed16 at ID/OOD. They beat their own references in only three of12 blocks (radius0.2/OOD for both seeds and radius0.5/OOD for seed15). Preserve these local gains without claiming all-radius actor added value.

## Next Step

Retain full-scope cubic-local feedback; stop load-only training and weight/seed sweeps. The next work is a read-only objective/critic diagnostic on the existing full-scope seed15/16 checkpoints, executed on servers. On the same paired initial states, compare actor versus fitted-reference first actions, followed by the same frozen actor for all remaining steps. Report both raw undiscounted cost and gamma0.99-discounted cost, and compare the continuation ordering with frozen critic mean/risk-adjusted action ordering. This tests whether the remaining gap reflects discount/evaluation mismatch or critic preference for a demonstrably worse first action. Separate ID-supported conclusions from OOD descriptions. No model fitting or optimizer updates; return only compact JSONs. This is implemented as v21; see `docs/nonlinear_load_v21_objective_protocol.md` for the frozen diagnostic and dispatch records.

## Limitations

Two seeds and one fully observed environment do not establish general RL or BAPR/RE-SAC superiority. Removing columns also reduces total penalty strength, so this is not an equal-total-regularization comparison. Initial-state alignment is state-matched; actor-trajectory alignment is not. Frozen critic ordering is not ground-truth value, and entropy/risk/reward scaling remain parts of the training objective. Direct references are feasible controls, not certified optima. Historical v13/v15/v17/v19 HOLD and paper tables remain unchanged.

Evidence: `results/nonlinear_load_v20_jacobian_scope/{task_observations,summary}.json` and the four per-scope/per-seed evaluation JSONs. Reproduce with `python3 scripts/summarize_nonlinear_load_v20.py`.
