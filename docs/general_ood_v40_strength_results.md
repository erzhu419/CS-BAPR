# v40 strength-matched result

Task `t135878` completed on node007 GPU0 with all three exits0, wall time2815 seconds (46.9 minutes). Each arm collected100000 interactions, performed99745 updates and restored100000 training context observations for final-checkpoint evaluation. Configs, runtime, training mode schedule, paired episode OOD seeds and all19 switch windows passed validation. Only1,985,807 bytes of JSON synced; checkpoints remain remote. Five summary regression tests passed.

| Arm | ID mean | Paired Markov50x mean | Worst fixed-profile mean | Non-ID transitions |
| --- | ---: | ---: | ---: | ---: |
| relative-adaptive | 1029.64 | 501.53 | 248.88 (heavy_traffic) | 3634 |
| fixed-minus2 | 1039.79 | 797.19 | 407.47 (mega_event) | 9389 |
| fixed-minus2p3 | 1032.63 | 637.83 | 259.86 (mega_event) | 6645 |

## Diagnosis

HOLD: adaptive loses to both constants on all three predeclared return measures. Its paired Markov difference is-295.66 versus fixed-minus2 (0/20 episode wins) and-136.30 versus fixed-minus2p3 (6/20 wins). Raising constant penalty from-2 to-2.3 also reduces OOD return, so strength explains part, but not all, of the adaptive deficit. The nearly equal average beta does not imply equal actor trajectories. Fixed-minus2p3's final1000-step alpha is25.12 versus9.30 adaptive and6.50 fixed-minus2; it is not a stability repair.

The adaptive beta averages-2.29311. Across17 actual profile changes, mean reset probability is0.020313 before the tick,0.020454 in its first50 transitions and0.020584 over the next500. At the two same-mode redraws, it is0.020553 before,0.021466 in the first50 and0.020381 over500. Post-switch reset peaks range0.02566-0.02852 for actual changes and0.02537-0.02745 for redraws. There is movement, but no convincing change-specific reset response beyond the0.02 prior in this diagnostic. Fifty-step post-tick beta means vary between-2.40466 and-2.21127, so this is not an exactly constant coefficient.

Reward z-score dominates98.123% of adaptive updates; Q-spread spikes dominate1.826%, relative critic drift0.051%. Source inspection corrects the original protocol description: reward/Q surprise uses the latest256-transition rollout, not the random replay batch. Replay reward variance still sets normalization; L1 drift uses global critic norms. Recent windows overlap by255 transitions, while the filter updates once per interaction and profile ticks are5000 interactions apart. This timing and normalization are the next hypotheses to test, not a demonstrated causal explanation of the return gap.

Adaptive and fixed-minus2 reproduce all previously recorded v39 training-block metrics and repaired evaluation results exactly. This confirms instrumentation preserved those trajectories, rather than supplying additional training-seed evidence.

## Next Step

Do not expand seeds, sweep beta/hazard or change manuscript claims. First use an existing frozen policy on paired actual-change and no-change environment runs, without optimizer updates, to test whether the current signals discriminate regime changes. Compare non-overlapping observation windows and a pre-window historical reference before connecting a revised signal to RL training. Keep perturbation profiles matched; changing them merely to enlarge a return gap would not establish adaptation.

## Limits

One previously selected seed and a100k development budget, not confirmation. The two redraws are descriptive controls; overlapping windows and within-block context history prevent treating episode/switch rows as independent training replicates. Belief updates in training only, not during50x OOD evaluation; extrema do not reveal exact sub-bin timing. Do not pool these results with v38's300k endpoints or attribute effects to absent NAU/SINDy/Jacobian mechanisms.

Machine-readable results: `results/general_ood_v40_strength/summary.json`.
