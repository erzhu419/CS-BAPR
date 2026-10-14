# v139 results and next step

Six formal tasks t145353--t145358 and two native smokes completed exit0.
All ran on Linux CPU: five formal probes on node007, one on jtl110gpu;
all three nodes eligible without preference/pin.3840 formal replay samples,
zero training interactions, optimizer updates and environment rollouts.
645332 formal JSON bytes,837838 including smokes pulled; no local snapshots.

Equal-sized age-stratum samples evaluated by final300k models:

| Case | Saved alpha | Behavior-action Q mean | Sampled TD MSE |
|---|---:|---:|---:|
| Friction seed3 | 555528 | 5.09e8 | 9.47e15 |
| Gravity seed4 | 1472593 | 1.18e9 | 2.20e18 |
| Friction seed4 | .02450 | 213.92 | 4.71 |
| Gravity seed3 | .01000 | 122.39 | .45 |
| ID seed3 | .03993 | 213.63 | 2.29 |
| ID seed4 | .08330 | 255.68 | 1.91 |

Divergent behavior-action values are already inflated, not only new-action Q.
Mean absolute bootstrap terms are5.54e8/1.32e9; entropy contributions are
only0.331%/0.340% of those magnitudes. Alpha-zero targets remain enormous.
No exact float32 tanh saturation observed. Epsilon-density differences persist
in tails (gravity seed4 next-state log-density gap max1.459), but the mean absolute
stable-target change is at most1.59e-7 of the divergent bootstrap magnitude.

Next: v140, critic lr3e-4 ->1e-4 only; fresh friction/gravity/ID seeds3/4,
same300k budget and evaluation. Planned, not dispatched. See
`docs/general_ood_v140_critic_rate_plan.md`; summary is
`results/general_ood_v139_frozen_diagnostics/summary.json`.

## Limitations
Late-model diagnostics do not identify the historical causal root. Sampling
overweights earlier collection intervals relative to their replay population;
these are not v138 training-block means. Removing divergence alone would not
resolve the stable controls' short OOD survival. Paper tables unchanged.
