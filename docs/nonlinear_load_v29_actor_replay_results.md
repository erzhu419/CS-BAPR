# Nonlinear load v29: actor early coverage regresses

Fifty focused/regression cases and GPU numerical preflight t130210 passed before formal dispatch.

Formal tasks t130215 (seed19) and t130216 (seed20) exited0 on automatically selected node007/GPU3 and jtl110gpu/GPU0 in77.80/136.58 minutes, including both sequential arms. All four evaluation JSONs were already synchronized (6,867,234 bytes); this analysis fetched no further remote artifacts. Frozen validation passed: shared prefix, runtime, native alpha/Jacobian and actor/critic sampling counts, matched controls, complete paired paths and physical-cost decomposition.

Formal accounting is398,000 actual training transitions/396,002 updates,22,000 shared identification/validation transitions and9,600 controller evaluation episodes (1,920,000 evaluation transitions). Both critics used early sampling; only actor Q/risk/entropy coverage differed. Last-checkpoint selection and all original gates remained fixed.

Percentage total-cost change, early actor minus uniform actor; positive is worse. Each cell uses100 paired paths.

| Radius | Load | Seed19 | Seed20 |
| --- | ---: | ---: | ---: |
| 0.2 | 1 | +0.311% | +0.341% |
| 0.2 | 50 | -0.018% | +0.234% |
| 0.5 | 1 | +0.543% | +0.540% |
| 0.5 | 50 | +0.194% | +0.366% |
| 1.0 | 1 | +1.167% | +1.121% |
| 1.0 | 50 | +0.801% | +0.879% |

Total cost worsens in11/12 blocks and early state cost worsens in12/12. At radius1/50x, state-cost increases of0.07835/0.09671 outweigh command-cost savings of0.01782/0.03064. Only36/19 of100 paired paths improve for seeds19/20. Initial output-force RMSE relative to fitted rises0.18489->0.28311 and0.15721->0.23536 on matched states. This is poorer initial regulation, not evidence that lower command effort improves the total objective.

The early actor loses to its own fitted reference in12/12 blocks; radius1 excess is2.970%/2.093% for seed19 and2.017%/1.570% for seed20 at ID/50x. Radius1 repair, small-radius retention and all-radius own-fitted added value all fail. The local v28 shadow-update advantage did not transfer to sustained online training.

## Next Step

Reject early actor sampling and retain the v25 early-critic/uniform-actor candidate. Stop actor sampling-ratio sweeps and seed/baseline expansion. Next propose a separately frozen environment-qualification diagnostic: assess recurring bounded load shocks, actuator saturation and continuing regulation costs before choosing a new training cohort. In these radius1/50x results, over99.5% of state cost occurs in the first50 steps; the current benchmark largely measures initial regulation despite repeated force refreshes. A new environment must test sustained disturbance handling while remaining controllable, rather than be selected for a favorable algorithm ranking. No follow-on jobs or manuscript changes in this analysis; historical HOLD remains unchanged.

## Limitations

Two training seeds and reused paths give descriptive evidence for this intervention, not general RL inferiority. The fitted controller is feasible, not certified optimal. Keep the v29 cohort separate from v25; shared seed labels are not independent replication. Changing the environment would create a new benchmark and cannot repair this negative result.

The user-authorized environment-qualification step completed as [v30](nonlinear_load_v30_environment_protocol.md): all three radii passed, with continuing cost dominated by command effort. See [v30 results](nonlinear_load_v30_environment_results.md). This supports a new learner pilot, not an algorithm-performance claim; the v29 rejection is unchanged.

Evidence: `experiments/nonlinear_load_v29_actor_replay_20261003/dispatch_manifest.json`; `results/nonlinear_load_v29_actor_replay/{task_observations,summary,block_analysis,preflight/summary}.json`; [frozen protocol](nonlinear_load_v29_actor_replay_protocol.md).
