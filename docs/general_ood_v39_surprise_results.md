# v39 surprise-scale results

Completed 2026-10-06: t135867 finished all four Hopper seed1 arms at100000 interactions/99745 updates, exit0, in about57 minutes. Evaluation-only task t135873 then reloaded the same final checkpoints with episode-seeded OOD randomness. Both tasks synced compact JSON; checkpoints stayed on node007. Reproduce the summary with `python3 scripts/summarize_general_ood_v39.py`.

| Arm | ID return | Paired Markov50x return | Worst fixed-profile return | Non-ID evaluation steps |
| --- | ---: | ---: | ---: | ---: |
| Raw adaptive | 210.20 | 210.20 | 193.20 | 0 |
| Relative adaptive | 1029.64 | 501.53 | 248.88 | 3634 |
| Fixed minus2 | 1039.79 | 797.19 | 407.47 | 9389 |
| Fixed minus3 | 210.20 | 210.20 | 193.20 | 0 |

The absolute critic L1-drift signal dominates99.996% of raw training updates. After warmup the raw policy uses beta=-3; its recorded penalty, alpha/Q/loss trajectories and all evaluation outcomes equal the fixed-minus3 control. Both show runaway training: final-block alpha166.05, Q-std4391.77 and critic loss1.55e10. Relative calibration removes this saturation: mean reset probability0.0205, window14.42 and beta-2.293; final alpha9.30/Q-std90.40, versus6.50/61.43 for fixed-minus2.

Calibration improves training and ID performance over the failed raw arm, but not over the matched fixed-minus2 baseline. Relative-minus-fixed2 differences are ID-10.14, paired Markov OOD-295.66 and fixed-profile floor-158.58. Its heavy_traffic return248.88 compares with fixed2's1049.57. The frozen advancement condition fails.

## Limitations

This is one deliberately selected development seed;20 evaluation episodes are not20 independent training seeds. Raw/fixed3 episodes terminate before the first OOD switch, so their Markov scores reflect ID failure, not exposed OOD adaptation. Original evaluation reset physical initial states but not the OOD RNG: variable episode lengths changed later episode schedules. The repaired evaluation preserves every ID/fixed-profile reward and length; only Markov scheduling changes (relative480.46 to501.53; fixed2 805.48 to797.19). Original JSON remains intact. The100k results are separate from v38's300k endpoints.

## Next Step

HOLD on confirmation expansion and manuscript superiority claims. Keep the relative-input repair as the development candidate and fixed-minus2 as its matched control. After the first1k steps, relative beta's1000-step means lie in[-2.310,-2.276], close to a constant rather than demonstrating useful timing. The next short diagnostic should add a strength-matched constant beta near-2.3 and switch-aligned short-window measurements before another large training round or environment retuning.
