# v98: equal-budget reference coverage comparison

Compare v97 old-recipe and diverse-recipe payloads against the unchanged v94 reference. Each recipe adds 4400 inputs/22000 known-mode labels: raw rows 23000 -> 27400, mean/error rows 25200 -> 29600, source groups/members 6 -> 8. Keep global forward/scales, 64-neighbor local ridge (.064), raw-mixture weight, disagreement coefficient (1), covariance formulas and temporal rules fixed. Reorder the mean support so the complete expanded raw support remains its prefix.

Native smoke checks both full payloads, bank/label alignment, 16 partial whole-seed-held-out errors, unchanged v94 scores and availability of all 95 evaluation paths. It does not produce a fitted candidate or calibration. Formal fitting recomputes all eight held-out error groups and calibrates all three covariance conditions on the identical 12000 normal-null transitions. Arrays/caches stay on node007; only scalar summaries are synced. No Windows placement or soft node/GPU preference.

Replay stage reuses the verified fit/calibration artifacts and evaluates 65 prior development paths plus all 30 v95 paths with complete 6000-step histories. Copy unchanged v94 controls and retain normal first800/full/last1000 windows, paired departure contrasts and first-post50/last64 mode/error attribution. Both recipes use identical reference budgets; no new environment steps or optimizer updates in v98.

## Limitations

One source policy and known simulator modes. The unchanged control has fewer rows/members; the two recipes have matched counts. All evaluated paths are development evidence. Reused calibration is not independent confirmation. Recognition/risk results are not actor adaptation or RL returns; fresh confirmation and actor integration remain pending.
