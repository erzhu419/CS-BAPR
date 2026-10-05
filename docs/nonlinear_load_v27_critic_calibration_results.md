# Nonlinear load v27: critic calibration does not transfer to 50x

t130155/t130156 (seeds19/20) completed with exit0 in 57.20/57.21 seconds on scheduler-selected Linux GPUs. Frozen validation passed. Formal calibration used 768 paired controller episodes, 153,600 simulator transitions and 2,000 optimizer updates; actors and original critics stayed unchanged. Evaluation reused v26 continuations and added no simulator transitions. Only 2,480,352 bytes of formal result JSON synchronized; raw labels and checkpoints remain remote.

Training direction MSE fell by 95.7%/92.7%, but held-out ID improvement was mixed and all four 50x direction discrepancies increased. The table reports mean absolute disagreement between predicted ensemble-mean Q derivatives and measured soft-Q derivatives at step0.005, with 100 paths and eight noise draws per block.

| Seed | Radius | Load | Original Error | Calibrated Error | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 19 | 0.2 | 50 | 0.127057 | 0.138389 | +8.9% |
| 19 | 1.0 | 50 | 0.082262 | 0.102239 | +24.3% |
| 20 | 0.2 | 50 | 0.108634 | 0.112242 | +3.3% |
| 20 | 1.0 | 50 | 0.087152 | 0.092664 | +6.3% |

Step0.01 gives the same conclusions. Small-radius 50x Q-sign disagreement counts improve slightly, whereas radius1 counts worsen; absolute-error deterioration does not mean every ordering metric deteriorates. At seed19/radius1/ID, the unpenalized actor derivative remains negative (-0.006010 -> -0.004991) against measured +0.003457. The key actor-score residual is not repaired.

## Decision And Next Step

Do not promote the calibrated critic. Retain the original v25 early models and historical HOLD; this frozen-actor experiment cannot establish a policy-return gain. Next, diagnose actor state coverage and full parameter-update directions with the original critic: compare uniform versus reset/early actor batches and attribute Q, risk, entropy and Jacobian contributions, including native Adam state. This is a proposed read-only diagnostic, not an established cause or a submitted job. No seed/baseline expansion or manuscript changes.

Follow-up v28 completed that diagnostic: both formal comparisons passed validation and actor early coverage gave better local physical-cost changes than uniform in every paired block. The effects are small and do not establish online performance gains. See [v28 results](nonlinear_load_v28_actor_update_results.md); no calibrated v27 critic is used.

## Limitations

Calibration used 48 initial states and four noise draws per seed; its pointwise value anchor does not constrain uncertainty derivatives everywhere. This rejects this fixed-budget calibration, not all Monte Carlo calibration. Local mean perturbations do not certify the complete actor gradient.

Evidence: `results/nonlinear_load_v27_critic_calibration/{summary,task_observations}.json`; frozen protocol: `docs/nonlinear_load_v27_critic_calibration_protocol.md`. Fourteen focused/regression tests passed, including exact checkpoint resume and unchanged controls.
