# v90 Matched coverage replay

Formal task `t138676` completed on node007 CPU, exit 0, in 5,462.20s (1h 31m 02s). All 65 paths and 96 changed early/tail windows are present. The three reused controls match v83 exactly; both additions used the same 12,000 normal-null transitions. Fixed-competitor covariance-volume terms remain unchanged. Only the 2,281,908-byte scalar JSON returned; traces and caches remain remote.

| Development readout | Negative recursive margins / 96 | Equal-window accuracy | Maximum normal mean risk |
| --- | ---: | ---: | ---: |
| v78 raw covariance | 14 | 84.54% | 0.007962 |
| v83 selected mixture | 12 | 87.06% | 0.007605 |
| Prefix mean addition | 9 | 89.50% | 0.005815 |
| Equal-count suffix addition | 12 | 86.68% | 0.006963 |

Prefix resolves four old negative windows and introduces one: v61 slippery early. Accuracy improves in 15 windows, regresses in six and is unchanged in 75. Suffix resolves none and introduces none; accuracy improves in nine and regresses in 17. All 96 paired departure-risk contrasts stay positive. Five of eight replay groups pass the existing screen under all three mixture readouts; v76/v62/v61 still fail.

- v62 tail accuracy: old 62.89%, prefix 81.45%, suffix 62.30%. Seed134020 heavy improves 18.75% -> 100%, high_grav 39.06% -> 100%; seed134021 heavy remains 26.56%.
- v61 tail: old 50.39%, prefix 66.80%, suffix 46.48%. Prefix retains heavy at 100% and repairs high_grav 0% -> 93.75%, but slippery remains 3.12% and compound regresses 100% -> 70.31%. Slippery early regresses 96% -> 44%, introducing a negative recursive margin.
- v76 prefix early/tail averages remain 56.50%/88.48%. Seed135180 high_grav early regresses 28% -> 14%, even though true-mode prediction RMS falls 0.3283 -> 0.1154. Slippery/compound early RMS also improves substantially without resolving negative mode evidence.

**Decision:** HOLD; retain v78 as the working baseline and both coverage additions as development evidence. Prefix coverage has localized value beyond an equal-count suffix addition, but is not a uniformly better readout. More accurate true-mode means do not necessarily improve relative mode ranking.

**Next:** a reference-only uncertainty-consistency pilot. Recompute whole-seed-held-out errors for each expanded mean model on the original 23,000 covariance-support rows; exclude all same-seed rows, including additions, from its predictor. Keep the raw covariance component, neighborhoods and v83 mixture weight fixed. Compare both additions with their frozen-error controls on the same six whole-seed reference holdouts before another replay. This tests whether old error distributions transport to the new means; it is a hypothesis, not an established explanation. No new seeds, actor feedback or follow-up task dispatched here.

## Limitations
These are previously inspected development paths from one source training seed, not independent confirmation or RL returns. Accuracy is an equal average over 48 changed paths and two windows, not pooled-transition accuracy. Covariances were intentionally frozen to isolate mean coverage, while both expanded means use 2,200 extra inputs and 11,000 cached labels. The historical v61 raw-covariance slippery tail accuracy of 87.50% remains an adverse comparison for the mixture family. Manuscript tables are unchanged.

Result: `results/general_ood_v90_coverage_replay/replay/coverage_replay.json`.
