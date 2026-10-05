# LQR feedback_v2 diagnostics

All 15 final checkpoints were analyzed on their existing workers, using CPU and two threads. No optimizer step or checkpoint write occurred. Only 420,907 bytes of diagnostic JSON were collected in `results/lqr_feedback_v2/diagnostics/`. Two focused tests pass, including exact parameter/optimizer-state preservation. CPU rollout returns reproduce the published evaluation returns with maximum relative difference 0.0000227.

CS-BAPR initial-state diagnostics below are averages of the three training-seed measurements on the 20 paired evaluation states. Action RMSE is per normalized-action component; Jacobian error is the mean squared Frobenius error.

| Amplitude | Action RMSE | Jacobian Error | Opposed Action Components |
| --- | ---: | ---: | ---: |
| 1x | 0.649 | 5.253 | 38.3% |
| 5x | 1.108 | 1.860 | 43.3% |
| 20x | 1.234 | 0.618 | 43.3% |
| 50x | 1.304 | 0.276 | 49.2% |

## Findings

The saved fitted feedback gain is correct to maximum absolute error 0.000000053. Its zero-state action is zero. CS-BAPR instead outputs `[0.604, -0.549]`, `[-0.299, -0.568]`, and `[-0.941, -0.738]` for seeds 0/1/2. The actor objective contains derivative matching but no action-value anchor. A regression test demonstrates that a constant action offset can have exactly zero Jacobian error.

On eight replay batches per checkpoint, the weighted Jacobian gradient's median norm is only 1.69%, 1.07%, and 1.28% of the SAC gradient norm. Their gradient cosine is negative in 9 of 24 batches. At 50x, 95% of reference action components and 58.3% of CS-BAPR components are saturated; the decreasing derivative error accompanies increasing action error.

The critic's inherited positive weight-regularization target shift is 13.88--15.60, versus normalized mean rewards of -0.25 to -0.33 and symbolic target shifts of 0.027--0.043. Critic pre-clipping gradient norms are approximately 11,814--15,123. This target-scale mismatch affects the shared backbone, not just the symbolic arm.

## Limitations

These are final-network snapshot gradients with frozen alpha and belief, not reconstructed historical optimizer updates or a causal ablation. Baselines without a fitted reference use known-model feedback only for diagnosis. Clipping multipliers are not Adam parameter-displacement measurements. No training result, environment, manuscript table, or author information was changed.

## Next Step

Prioritize a separately frozen action-value-anchor ablation, keeping the environment, seeds and training budget unchanged. Evaluate ID action bias and control cost as well as OOD return. Treat backbone target-scale correction as a separate comparison rather than changing both mechanisms at once. Raw measurements and per-seed summaries are retained; `scripts/diagnose_lqr_feedback_v2.py` reproduces each checkpoint analysis.

The next round is implemented and dispatched under `docs/lqr_action_anchor_v3_protocol.md`: twelve fresh runs, fixed anchor weight 1.0, and unchanged critic settings. The original 25-test suite and a three-case diagnostic follow-up pass; the CUDA six-episode startup/evaluation preflight completes. Task IDs are recorded in `experiments/lqr_action_anchor_v3_20260930/dispatch_manifest.json`.
