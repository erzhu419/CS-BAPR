# Nonlinear load v26: early sampling improves local Q ordering

t130081/t130082 exited 0 in 6.69/7.81 minutes on scheduler-selected node007 GPUs3/0. All 64,000 stochastic and 1,600 deterministic reproduction episodes passed frozen validation. Parameters stayed unchanged; fitting/optimizer updates were zero. Compact results total 3,538,064 bytes, automatically synchronized; 23,744,591 bytes of raw matrices remain remote. No checkpoint pull, new jobs or manuscript changes.

At radius1, finite-difference step0.005, each block has 100 paths and eight noise draws. Positive return/score derivatives favor the direction toward the fitted mean.

| Seed | Load | Early Measured Soft Q | Early Predicted Q Mean | Q Sign Disagreements: Uniform -> Early |
| --- | --- | ---: | ---: | ---: |
| 19 | 1 | +0.012594 | +0.003127 | 66 -> 33 |
| 19 | 50 | +0.012281 | +0.033579 | 64 -> 38 |
| 20 | 1 | +0.012901 | +0.011514 | 62 -> 11 |
| 20 | 50 | +0.013415 | +0.065029 | 64 -> 21 |

Early Q and measured soft-Q block means are positive in all eight draws at both step sizes in these four blocks. Model-own Q-direction absolute discrepancy falls in all eight seed/radius/load blocks. Measured return-path signs never reverse between steps0.01/0.005.

Remaining actor-score discrepancies matter. At seed19/radius1/ID, measured soft actor derivative is +0.003457 versus unpenalized prediction -0.006010; the risk term changes the prediction to +0.004122. Adding risk worsens early actor-score path ordering in the other three radius1 blocks, so uniform risk removal is not supported. Small-radius Q sign disagreements change 39->45,45->44,39->24,40->42; moving toward the fitted reference also worsens discounted physical cost in three of four early small-radius blocks.

## Next Step

Retain early sampling. Propose one frozen-actor, critic-only calibration experiment using independent ID paths and paired soft-Q return differences as direction labels. Keep each seed's actor, reference, terminal alpha/reward scale and risk coefficient fixed; include small-radius states, allow either label sign and keep 50x exclusively for evaluation. Compare the original critic with its calibrated clone before another online training run. No seed/baseline expansion or follow-on tasks are submitted here.

Follow-up v27 completed this fixed-budget calibration. Training direction loss fell, but all four 50x direction absolute discrepancies increased and the key seed19 ID actor-score residual remained. The calibrated critic is not promoted; see [v27 results](nonlinear_load_v27_critic_calibration_results.md). The next proposed diagnostic concerns actor state coverage and complete parameter-update directions.

## Limitations

Each model has its own actor, reward scale and unit direction; these are local ordering comparisons, not same-action causal error certificates or full parameter-gradient results. Reused paths and two seeds do not establish general RL superiority. This read-only diagnostic does not change v25's failed all-radius own-fitted gate; historical HOLD remains unchanged.

Evidence: `results/nonlinear_load_v26_critic_direction_diagnostic/{summary,block_analysis,task_observations}.json`; validator and comparison: `scripts/summarize_nonlinear_load_v26.py` (five focused tests passed).
