# Nonlinear load v28: actor coverage improves the local update

t130187/t130188 exited0 in5.89/4.92 minutes on scheduler-selected node007/GPU3 and jtl110gpu/GPU0. Formal validation passed, including all v26 unperturbed baselines, paired paths/noise and return decompositions. The57,600 episodes used11,520,000 evaluation transitions and258 disposable Adam steps, with zero training transitions/retained optimizer updates. Original actors and critics stayed unchanged. Fifteen focused/regression tests and the numerical preflight passed.

At parameter scale1, the table reports the change in mean discounted physical cost relative to the unperturbed actor, in units of0.001; negative is better. Each arm's direction is the mean of64 independently restored, native-state Adam displacements, not a trained policy or the Adam step of an averaged gradient.

| Seed | Radius | Load | Uniform Change | Early Change |
| --- | ---: | ---: | ---: | ---: |
| 19 | 0.2 | 1 | +1.579 | +0.493 |
| 19 | 0.2 | 50 | +1.086 | +0.038 |
| 19 | 1.0 | 1 | +0.949 | -0.057 |
| 19 | 1.0 | 50 | +0.459 | -0.509 |
| 20 | 0.2 | 1 | -3.336 | -4.743 |
| 20 | 0.2 | 50 | -3.457 | -4.844 |
| 20 | 1.0 | 1 | -3.648 | -5.046 |
| 20 | 1.0 | 50 | -3.779 | -5.160 |

Early has lower cost than uniform in all16 seed/radius/load/scale blocks, each across8/8 noise-draw means and100/100 noise-averaged paths. Both scales agree: early improves on the unperturbed mean in6/8 seed/radius/load blocks versus4/8 for uniform. Seed19 small-radius cost still increases under early sampling. At radius1/50x, seed19 early decreases cost in8/8 draws versus1/8 under uniform. The single-step changes relative to baseline remain below0.04% in magnitude.

The uniform batches contain1,547/1,507 late states out of2,048 for seeds19/20 and only6/16 resets. Increasing actor reset/early coverage therefore affects the measured complete-parameter update. However, its direction is close to uniform (cosines0.9967/0.9993), and both arms align strongly with the zero-data-gradient Adam step (cosines0.9873--0.9979), which includes momentum and weight decay. Native optimizer history matters; this is not an additive decomposition of Adam.

Soft return is not physical cost: seed19/radius0.2/50x has positive soft-return derivatives while cost increases. Initial critic scores also oppose measured soft-return improvement for seed20 at both50x radii, and for seed19 early/radius1/50x. Do not use surrogate-score improvement as the physical-performance gate.

## Decision And Next Step

This supports one online actor-sampling comparison, not promotion or learner replication. Propose v29 with the same seeds19/20, a shared complete warmup prefix and500 episodes per arm: both critics use the retained v25 early sampler, while only actor Q/risk/entropy coverage changes from uniform to50% uniform/25% reset/25% early. Keep the native alpha/reward/belief batches, reset-balanced Jacobian inputs, reference, loss weights, optimizer history and all other rules matched; subsequent learning differences are consequences of the intervention. Evaluate last checkpoints at the original three radii, ID/50x and100 paths using the existing own-fitted/cost gates. No new tasks are submitted in this analysis; historical HOLD and manuscript remain unchanged.

Follow-up v29 completed and passed frozen validation, but early actor coverage worsened total cost in11/12 blocks and failed every admission gate. See [v29 results](nonlinear_load_v29_actor_replay_results.md); the local shadow-update advantage did not transfer to online training. Reject this actor-sampling intervention.

## Limitations

This measures local averaged shadow updates at two terminal checkpoints on reused paths, not sustained online learning or a full actor derivative certificate. The small physical effects and remaining seed19 regression do not establish bus-like performance or general RL superiority. Direction construction uses ID replay only; OOD outcomes are evaluation evidence.

Formal compact JSON totals1,889,230 bytes;12,660,880 bytes of raw matrices remain remote, as do full replay/checkpoints. The preceding input export pulled only2,123,886 bytes. Evidence: `results/nonlinear_load_v28_actor_update_diagnostic/{summary,block_analysis,task_observations}.json`; [frozen protocol](nonlinear_load_v28_actor_update_protocol.md).
