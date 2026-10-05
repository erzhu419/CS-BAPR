# RandomLoad v6: completed results

All 15 tasks (t116789--t116803) finished with exit code 0. Every final evaluation verifies the frozen configuration, 500 episodes, 100,000 learner steps, 99,001 updates and 20 paired evaluation seeds (10000--10019) at each severity. The 15 evaluation JSON files total 403,555 bytes; checkpoints were not downloaded. The three missing local-worker results were recovered directly from local output directories.

Cost is minus episode return; lower is better. Means and sample SDs are over three training-seed means, not 60 independent trained models.

| Method | 1x | 2x | 5x | 10x | 20x | 50x | 50x Seed SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CS-BAPR | 46.34 | 46.34 | 46.34 | 46.35 | 46.44 | 48.28 | 31.96 |
| BAPR | 53.00 | 53.01 | 58.91 | 77.54 | 122.04 | 907.57 | 437.72 |
| RE-SAC | 43.76 | 44.86 | 44.94 | 56.90 | 101.91 | 529.58 | 230.06 |
| No SINDy | 41.40 | 41.43 | 41.44 | 41.46 | 41.83 | 45.06 | 38.49 |
| No NAU | 16.44 | 16.44 | 16.45 | 16.47 | 16.63 | 18.25 | 27.01 |

CS-BAPR reduces 20x/50x mean cost by 61.95%/94.68% versus BAPR and 54.43%/90.88% versus RE-SAC, winning all three matched training seeds at both severities. Its mean cost increases only 1.94 between 1x and 50x, versus 854.57 for BAPR and 485.83 for RE-SAC. ID mean cost is nevertheless 5.90% higher than RE-SAC's. The ongoing-load stress test therefore gives a positive baseline-relative OOD result, not uniform superiority.

The component result is negative for the full architecture. No-NAU beats full CS-BAPR at every severity on every training seed; its 50x cost is 62.21% lower. No-SINDy has lower mean cost at every severity; full CS-BAPR costs 7.16% more at 50x, although it wins two of the three matched seeds. This does not establish either bundle's necessity. No-SINDy removes both the symbolic critic target penalty and actor Jacobian alignment (and identification data); no-NAU changes the actor architecture and arithmetic regularization together.

## Diagnostic Evidence

Paired non-learning controls were recomputed from the frozen environment on the same formal evaluation seeds, without learned checkpoints:

| Control | 1x Cost | 20x Cost | 50x Cost |
| --- | ---: | ---: | ---: |
| Zero action | 104.17 | 940.20 | 6751.67 |
| Force cancellation only | 101.85 | 101.98 | 102.92 |
| Physical feedback, force-blind | 0.14 | 0.41 | 2.34 |
| Physical feedback plus force compensation | 0.14 | 0.27 | 1.20 |

The compensated reference has zero saturated components throughout all control rollouts. Its low cost shows that fixed control authority can handle these particular sampled disturbances. Full CS-BAPR remains far from this feasible feedback behavior even at ID.

Full CS-BAPR's zero-observation deterministic actions are `[-0.234,-0.725]`, `[-0.465,0.234]`, and `[0.497,0.457]`; the fitted feedback maps zero to zero. At 50x, initial-observation squared-Frobenius Jacobian error averages 0.367 for full CS-BAPR, versus 30.615 for BAPR, 24.509 for RE-SAC, 7.987 for no-SINDy and 0.170 for no-NAU. A closer derivative therefore coexists with substantial action offset. The offset is a plausible bottleneck, not yet a demonstrated cause; the saved alignment metrics cover reset observations, not learned-policy trajectories.

## Limitations

This is observed stochastic load on fixed linear dynamics, not a standard Gym benchmark, hidden-regime adaptation or bus-level generality. Only three training seeds; no significance claim. Symbolic arms additionally collect 11,000 identification/validation transitions: 111,000 total non-evaluation interactions versus 100,000 for other arms. Baseline stabilization/behavior-cloning presets differ from CS-family regularization. Local CS/BAPR seed 0 and RE-SAC seed 2 used Torch 2.8.0+cu128; other runs used 2.5.1+cu121. Paired seeds do not remove this runtime difference. The known-model tanh feedback is a privileged non-learning reference, not a constrained/stochastic optimum. Raw v5 and v6 costs are not directly comparable. Prior negative results and manuscript tables remain unchanged.

## Next Step

Keep the environment, loss weights and training budget fixed; do not expand training seeds yet. Diagnose the nine full/no-NAU/no-SINDy final checkpoints on their existing workers, with no optimizer updates or checkpoint transfers. First reproduce the saved returns. Then compare original actions with the post-hoc counterfactual `tanh(mu(s)-mu(0))` on the same paired seeds at 1x/20x/50x. Measure trajectory-state action error, physical-state versus force-input Jacobian error, saturation and physical-state drift; evaluate saved fitted feedback for symbolic arms as a separate control. Report this as diagnosis, not a newly trained method. A consistent reduction of both ID and OOD cost would justify a separately frozen equilibrium-preserving training experiment; otherwise inspect state-dependent actor/critic gradients rather than tuning the environment to favor a method.

Reproduce: `experiments/random_load_v6_20261001/runtime/.runtime/bin/python scripts/summarize_random_load_v6.py`. Evidence: `results/random_load_v6/summary.json`, `seed_mean_returns.csv`, `task_observations.json` and all 15 per-run evaluation JSONs.
