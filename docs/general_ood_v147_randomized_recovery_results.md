# v147 randomized recovery-training results

All four formal tasks t151879--t151882 exited 0. Each completed 600,000
interactions and 599,745 updates, without resume. Both predeclared snapshots
were retained: 1,440 CPU evaluation episodes, 1,280 paired prefixes verified.
Only 12,164,333 bytes of scalar JSON were pulled; checkpoints/replay remain
server-side. Eleven summary tests passed.

Final 600k sampled results: mean episode length / healthy 1,000-step finishes,
15 common-reset/noise episodes per cell. Pulses use the held-out 2.659x gravity.

| Training | Seed | No pulse | After100 pulse10 | After300 pulse10 | After300 pulse25 |
|---|---:|---:|---:|---:|---:|
| No-pulse | 5 | 1000 / 15 | 191 / 0 | 975 / 14 | 363 / 0 |
| Randomized recovery | 5 | 509 / 0 | 264 / 0 | 457 / 0 | 399 / 0 |
| No-pulse | 6 | 1000 / 15 | 305 / 1 | 1000 / 15 | 777 / 9 |
| Randomized recovery | 6 | 1000 / 15 | 1000 / 15 | 877 / 11 | 389 / 0 |

Seed6 gains substantially on the early ten-step pulse, but loses on the later
25-step pulse. Seed5 loses normal-environment stability; deterministic normal
evaluation also falls (mean616, 0/5 full finishes), so action noise alone does
not explain it. Seed5 randomized policies pass100-step recovery in all15
early/later ten-step pulse episodes but subsequently all fall. No stable
25/50-step or permanent-peak advantage was established. Both300k snapshots
are retained in the summary, rather than selecting a favorable checkpoint.

Training exposure was sparse: seed5 had987 assigned/466 exposed/440 restored
pulse episodes, with4,550 peak steps (0.758% of training); seed6 had
1,183/500/467, with4,824 peak steps (0.804%). Half-episode assignment does not
mean half the interactions were disturbed; early falls often precede onset.

Next: v148 frozen evaluation of the training-known2.0x ten-step pulse at
onsets100/300. Native smoke exposed cross-host roundoff (about1e-14, identical
policy observations), not a v147 data error. No-pulse and2.659x controls are
evaluated in the same run for exact pairs. Separate
within-support recovery failure from amplitude extrapolation before training.

## Limitations

Two training seeds show heterogeneous outcomes, not stable robustness or a
CS-BAPR contribution. Reset/noise repeats are not independent training seeds.
