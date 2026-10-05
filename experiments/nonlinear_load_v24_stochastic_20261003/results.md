# Nonlinear load v24: stochastic alignment leaves a critic-ranking discrepancy

Formal t128953 exited0 in341.2 seconds; all32,000 controller episodes passed the frozen validator. Parameters stayed unchanged, with zero fitting/optimizer updates. The formal compact JSON is1,781,921 bytes; the11,981,564-byte raw draw/path matrices remain remote. No new jobs or manuscript changes in this analysis.

## Findings

Radius1 means over100 paired paths and eight common action-noise draws, finite-difference step0.005 toward the fitted latent mean. These are directional derivatives, not percentage gains; positive return/score is preferred. The actor prediction below excludes the risk term and includes initial entropy.

| Model | Load | Measured Soft Q | Predicted Q Mean | Measured Soft Actor | Predicted Unpenalized Actor |
| --- | --- | ---: | ---: | ---: | ---: |
| floor | 1 | +0.012854 | -0.041465 | +0.009268 | -0.045050 |
| no-floor | 1 | +0.010902 | -0.039619 | +0.010494 | -0.040028 |
| floor | 50 | +0.012924 | +0.089674 | +0.008988 | +0.085738 |
| no-floor | 50 | +0.010651 | -0.054849 | +0.010211 | -0.055289 |

All eight individual draw means retain these signs in each radius1 block. Matching stochastic continuation, native reward scaling and entropy timing therefore does not remove the discrepancy in three of four blocks. Removing the risk term alone does not repair the frozen ID predictions. Floor/50x agrees in block-mean sign, so this is not a universal OOD failure. Q mean opposes56/77 and59/82 improving ID paths for floor/no-floor; even floor/50x opposes41/77.

At radius1, discounted physical reward and future entropy both improve; initial entropy falls but the net measured actor return remains positive. Physical improvement is predominantly early state regulation. Radius0.2 is mixed: floor/ID physical cost worsens and its soft Q falls; floor/50x raw cost rises while discounted cost falls. No measured discounted-cost, soft-Q or soft-actor path sign reverses between steps0.01/0.005. At radius1, Q-mean finite differences and analytic derivatives agree in path sign throughout.

## Next Step

Prioritize a critic-learning intervention over another entropy/risk sweep. The proposed single-variable comparison enriches only the critic minibatch with reset/early transitions, keeping actor/alpha/belief/Jacobian batches, floor, reference, environment and interaction/update budgets unchanged. This differs from v15's wider environment resets and v17's Jacobian-state sampling. Critic coverage remains a hypothesis, not a demonstrated cause in seed18; freeze the sampling rule and matched control before dispatch. No follow-on training is admitted by this diagnostic; historical HOLD remains unchanged.

The user-authorized next step is completed as v25 with a shared full warmup checkpoint and seeds19/20. Paired formal tasks t129046/t129047 passed frozen validation; radius1 repair and small-radius retention pass, but all-radius own-fitted added value fails. See `docs/nonlinear_load_v25_critic_replay_results.md`; historical HOLD remains unchanged.

## Limitations

This finite-horizon snapshot freezes alpha/reward scale and does not reconstruct nonstationary training or the pessimistic min-ensemble fixed point. Eight noise draws, one reused training seed and reused evaluation paths provide descriptive local evidence, not a critic-error certificate, full actor-gradient result or general performance claim. Evidence: `results/nonlinear_load_v24_stochastic_diagnostic/summary.json`; validator: `scripts/summarize_nonlinear_load_v24.py` under the frozen runtime.
