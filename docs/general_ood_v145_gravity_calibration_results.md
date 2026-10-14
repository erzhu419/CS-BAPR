# v145 gravity calibration results

All four final cases completed. Seed3 attempts t151458/t151459 stopped because
the jtl110gpu root filesystem filled while SB3 created a per-block /tmp logger.
t151579/t151580 restored the complete275k states and reached600k; seed4 tasks
t151460/t151461 completed directly. Retained all120 diagnostic blocks and
all24 periodic checks per case; cumulative2.4M interactions /2398980 updates.
320 CPU episodes /90613 steps;1540407 JSON bytes pulled, no models/replay.

| Training gravity | Seed | Own sampled length300k ->600k | Peak2.659 sampled length300k ->600k |
|---|---:|---:|---:|
| 1.5 | 3 | 596.47 ->737.20 | 51.13 ->93.60 |
| 1.5 | 4 | 1000 ->1000 | 88.93 ->95.67 |
| 2.0 | 3 | 105.33 ->104.07 | 85.53 ->87.27 |
| 2.0 | 4 | 93.73 ->225.13 | 80.00 ->132.53 |

Each cell averages15 common-noise episodes. At600k,1.5 seed3 finishes10/20
own-domain episodes and seed4 finishes20/20;2.0 policies finish none. All80
held-out peak episodes at600k fall. Do not select isolated successful periodic
checkpoints: seed4 at2.0 briefly reaches1000 during500k periodic evaluation,
but the predeclared600k snapshot remains unstable.

Next v146: frozen CPU pulses at the unchanged2.659 peak, then restore baseline
gravity. Compare ID and1.5 policies,100/300-step onset times,10/25/50-step
durations, permanent peak and no-pulse controls. No state reset at switches;
measure100-step recovery and report episodes that fail before exposure.
The root backend now reuses one task-local SB3 logger; archived runtimes stay
unchanged. jtl110gpu root still has only15MB free; /data remains spacious.

## Limitations
Two training seeds, with cross-node full-state resumes for seed3. This is
gravity calibration/frozen transfer, not CS-BAPR adaptation or impossibility
evidence.1.5 seed3 instability and all peak failures are retained.
