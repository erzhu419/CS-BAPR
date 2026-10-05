# Nonlinear load v25: consistent repair, no all-radius added value

Paired tasks t129046/t129047 exited0 in71.6/77.1 minutes on scheduler-selected node007 GPUs0/1. All four arm results passed frozen validation:398,000 actual training transitions,396,002 updates,22,000 shared identification/validation interactions and9,600 controller evaluation episodes. Formal JSONs total6,864,502 bytes; only the two missing seed20 JSONs were fetched (3,432,907 bytes). No checkpoint/replay pull, new jobs or manuscript changes.

Radius1 changes after early-enriched critic sampling relative to same-seed uniform control; negative is better. Each block has100 paired paths.

| Seed | Load | Total Cost Change | Early State Cost Change | Improving Paths |
| --- | --- | ---: | ---: | ---: |
| 19 | 1 | -2.012% | -2.892% | 88/100 |
| 19 | 50 | -1.380% | -2.922% | 83/100 |
| 20 | 1 | -3.040% | -5.840% | 88/100 |
| 20 | 50 | -2.315% | -5.818% | 94/100 |

Total and early-state means improve in all12 seed/radius/load blocks. Radius0.2 total cost falls0.076--0.514%. Radius1 regulation gains outweigh increased command cost. The radius1 repair and small-radius retention gates pass; own-fitted added value fails. The early actor beats its fitted reference in only3/12 blocks, all outside radius1; radius1 excess remains1.782%/1.283% for seed19 and0.819%/0.520% for seed20 at ID/50x. Learner replication remains unadmitted; historical HOLD is unchanged.

At matched radius1/ID initial states, physical-input action-Jacobian squared error relative to the fitted reference falls0.09881->0.02188 and0.09755->0.00844 for seeds19/20. Conversely, actor load-cancellation residual rises0.07710->0.11553 and0.11019->0.12762, also rising in both50x blocks. The intervention brings the actor closer to its reference; improved cost does not demonstrate corrected critic ordering or better local load cancellation.

## Next Step

Retain early sampling as the next candidate, with the floor/full Jacobian/cubic-local reference unchanged. Freeze both seeds' uniform/early last models and reuse v24's paired stochastic soft-return direction diagnostic, separating Q mean and risk-adjusted scores under each model's terminal alpha/reward scale. Choose one further intervention only after that comparison; no sampling-ratio sweep, seed expansion or BAPR/RE-SAC dispatch in this analysis.

This diagnostic is complete as v26: both source seeds passed validation, with improved radius1 local Q ordering but remaining actor-score and small-radius discrepancies. See `docs/nonlinear_load_v26_critic_direction_results.md` for the proposed frozen-actor critic calibration; historical HOLD remains unchanged.

## Limitations

Two training seeds, reused evaluation paths and one fully observed environment do not establish general RL superiority. Mean improvement is not universal path improvement (seed19/radius0.2/50x improves45/100 paths). Local cancellation residuals do not identify the cause of the remaining cost gap; fitted references are feasible controls, not certified optima.

Evidence: `results/nonlinear_load_v25_critic_replay/{summary,block_analysis,task_observations}.json` and four per-arm evaluation JSONs. Frozen validator: `scripts/summarize_nonlinear_load_v25.py` under the pinned runtime.
