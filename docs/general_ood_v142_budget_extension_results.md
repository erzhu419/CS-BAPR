# v142 results and next step

All six tasks t147938--t147943 completed exit0.1.8M new interactions/updates;
cumulative3.6M interactions /3598470 updates. Configurations, full-state lineage,
all60 prefix diagnostic blocks and12 prefix evaluations match v141 exactly.
The300k boundary evaluation is present.1138628 scalar JSON bytes pulled;
snapshots/replay remain on servers. Training/continuation took107.2--122.9 min.

| Profile | Seed3 sampled length300k ->600k /15 | Seed4 sampled length300k ->600k /15 |
|---|---:|---:|
| Friction | 136.33 ->125.00 | 146.13 ->152.67 |
| Gravity | 93.13 ->89.60 | 87.40 ->88.87 |
| ID | 139.80 ->146.73 | 1000 ->1000 |

Final120 CPU episodes /32067 steps:100 falls,20 time limits (ID4 only).
All trajectories remain numerically moderate; final5k TD MSE.377--5.026.
ID3 remains near145 in late periodic evaluations. Extra budget neither restores
ID3 nor breaks the OOD survival plateaus; stop this learning-rate/budget route.

Existing endpoint logs already localize failure:95 height-only and5 height+angle
terminations, no state-bound violations. All100 failures cross the native.7m
height boundary. This is posture/control failure, not an extreme-velocity exit;
the actual action/torque history is still missing.

Next v143: zero-training CPU terminal-window traces for all six final models,
plus stable ID4 frozen on friction/gravity as positive control. No reward,
healthy-threshold or physics changes. Summary: `results/general_ood_v142_budget_extension/summary.json`.

## Limitations
Two training seeds and within-trajectory continuation. Stable ID4 versus failed
ID3 in identical physics prevents attributing all failure to OOD severity.
These results do not establish physical impossibility or baseline superiority.
