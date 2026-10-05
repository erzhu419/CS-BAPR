# LQR action_anchor_v3 results

All 12 manifest tasks finished with exit code 0. Their local final-evaluation JSON files verify 500 episodes, 100,000 environment steps, 99,001 updates, and 20 paired evaluation seeds per amplitude. The anchored/original CS-BAPR configurations differ only in action-anchor weight (1 versus 0). No checkpoint was downloaded; the source JSON files total 304,531 bytes.

Mean episodic return in thousands, averaged over three training seeds; higher is better. Individual seed means, training-seed sample deviations, action diagnostics and runtime versions are retained in `results/lqr_action_anchor_v3/summary.json`; `seed_mean_returns.csv` contains the 72 seed/amplitude means.

| Method | 1x | 2x | 5x | 10x | 20x | 50x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anchored CS-BAPR | -276.806 | -299.143 | -307.580 | -677.544 | -1504.323 | -5552.749 |
| Original CS-BAPR | -2.506 | -3.025 | -247.167 | -1316.313 | -3043.864 | -9591.081 |
| BAPR | -779.405 | -811.327 | -944.273 | -1753.611 | -2915.716 | -9591.179 |
| RE-SAC | -0.439 | -0.486 | -227.734 | -2215.584 | -4356.997 | -11951.601 |

## Findings

- At 20x/50x, anchoring wins all three matched training-seed comparisons against each learned comparator. Mean cost reductions versus original CS-BAPR are 50.6%/42.1%, versus BAPR 48.4%/42.1%, and versus RE-SAC 65.5%/53.5%. At 10x, it wins 2/3 against original CS-BAPR and 3/3 against each baseline.
- At 1x/2x, anchoring loses every seed comparison against original CS-BAPR and RE-SAC. Its mean costs are 110.5/98.9 times original CS-BAPR's. At 5x, mean cost increases 24.4% versus original CS-BAPR and 35.1% versus RE-SAC; only seed 0 wins these comparisons.
- Initial-state action-component RMSE at 50x drops from 1.360 to 1.100, and opposed-action fraction from 54.2% to 36.7%, averaged over training seeds. At 1x, RMSE instead rises from 0.838 to 0.920. The action anchor has not restored the equilibrium: seed 2's zero-state action is `[0.977, -0.788]` and its ID evaluation return is -786,540.
- Previously computed paired zero-action controls remain applicable because the frozen environment sources and evaluation seeds match. At 20x, anchored cost 1,504,323 still exceeds zero-action cost 1,018,450; at 50x it improves on zero action by 12.8%. Known-model tanh feedback costs 20,971/1,234,719 at 20x/50x, substantially below the learned policy.

## Limitations

This remains fixed-linear-dynamics initial-state extrapolation, with a task-specific fitted-model teacher and three training seeds, not bus-style randomness or general RL validation. All runs use Torch 2.5.1+cu121 but GPU/Python versions differ. Baselines inherit project stabilization. Reused controls are privileged or non-learning diagnostics, not new trained comparators. No statistical-significance claim is made.

## Next Step

HOLD: retain the high-amplitude gains and ID failures; do not replace manuscript tables or expand seeds. The six-checkpoint read-only diagnosis is complete in `docs/lqr_action_anchor_v3_diagnostics.md`. It rules out uniformly negligible anchor gradients and identifies a persistent shared critic target-scale mismatch. The isolated target-shift ablation is dispatched separately under `docs/lqr_critic_shift_v4_protocol.md`; this completed v3 study remains unchanged.

Reproduce the local aggregation with `python3 scripts/summarize_lqr_action_anchor_v3.py`.
