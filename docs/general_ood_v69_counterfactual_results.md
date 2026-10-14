# v69 result and next step

Task t136910 completed on node007 GPU1 in 402.3 s, exit 0. Six local tests
passed. All 24,000 production normal-shadow replays were exact. The reference
ledger contains 156,000 simulator calls and 18,000 normal reference updates;
102,000 saved transitions were replayed. Only the 450 KB scalar JSON was pulled.

| Group | Normal null / maximum mean | Positive first50 / tail | Positive true-mode margins first50 / tail |
| --- | --- | --- | --- |
| v62 | pass / 0.0256 | 8/8, 8/8 | 2/8, 0/8 |
| v61 development | pass / 0.0408 | 2/4, 2/4 | 0/4, 0/4 |

The v62 gravity first50 contrasts are +0.0840 and +0.0542, improving on v66's
early failure. Some v62 tail contrasts remain very small. Old gravity and
compound paths still fail both detection windows. All changed v62 tails are
classified as normal; 3/4 old changed tails are also entirely normal.

Matched normal-path density-calibration mode accuracy is 98-100%. Its contrast
with poor shifted-path identification points to unresolved state-path
generalization. Keep v66 as control and v69 as a partial detector result;
overall HOLD, with no belief/controller integration or manuscript changes.

Next: audit true-mode full-state replay and physical template separation on
fresh gravity/compound paths, including quiet tails. Distinguish model/state
coverage failure from weak physical identifiability before expanding references.

## Limitations

Calibration diagnostics reuse density-fitting data and are not confirmation.
The screen adds simulator/mode access and updating-policy coverage, reuses old
evaluation paths, and has one source training seed. Standalone verification
used another 8,000 simulator calls outside the reference ledger; two failed
starts did not log their pre-checkpoint counts. The warm-start and stale-script
deployment problems were repaired before the completed run. No RL-return gain
is established.
