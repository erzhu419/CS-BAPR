# LQR action_anchor_v3 diagnostics

The six anchored/original final checkpoints were analyzed on their workers with CPU/two threads, without optimizer steps or checkpoint writes. Only 196,852 bytes of diagnostic JSON were pulled. Four focused tests pass. CPU versus original GPU returns differ by at most 0.01584% per episode and 0.0001061% per amplitude mean. Raw measurements and per-seed medians are in `results/lqr_action_anchor_v3/diagnostics/summary.json`.

## Findings

- Fitted feedback gains remain correct: maximum absolute error 0.000000053. Identification is not the current bottleneck.
- Anchored replay unit-ball coverage is 11.70%, 6.00%, 9.18% for seeds 0/1/2; original coverage is 34.15%, 8.72%, 3.02%. Training starts in the unit ball but trajectories leave it. Reduced ID coverage is not a consistent explanation across all three paired runs.
- Weighted anchor/SAC gradient-norm ratios have batch medians 21.1%, 33.3%, 11.1%. Gradient cosines are negative in 5/8, 4/8, 0/8 batches. The worst ID seed therefore does not exhibit conflicting final-snapshot gradients. Its replay action RMSE is still 1.154; anchoring has not learned the reference accurately even on sampled replay states.
- Anchored positive weight-regularization shifts added to next-state Q are 22.40, 16.35, 20.45, versus normalized mean rewards -0.068, -0.373, -0.400. Critic pre-clipping gradient norms are 19,154, 31,828, 485,521; original norms are 12,621, 15,114, 11,999. The shared target-scale mismatch persists and is especially pronounced in anchored seed 2.

## Limitations

These are eight final-network replay batches per checkpoint, with frozen alpha/belief, not historical update reconstruction or causal attribution. Gradient clipping is not an Adam displacement measurement. No training, environment or manuscript result was changed.

## Next Step

Prioritize an isolated 2-by-2 comparison: action anchor 0/1 crossed with the positive critic target-shift coefficient at its current value/zero, keeping all other settings fixed. Check ID stability before judging OOD gains. Do not increase the anchor coefficient or OOD amplitude based on this snapshot. The comparison is implemented and dispatched separately under `docs/lqr_critic_shift_v4_protocol.md`; all twelve tasks reached training on 2026-10-01.
