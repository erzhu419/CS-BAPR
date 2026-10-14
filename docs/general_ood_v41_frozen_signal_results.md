# v41 frozen-signal result

Task `t135894` completed on node007 GPU0 in287.5 seconds (4.8 minutes). Both existing v40 policies completed100 paired rollouts each:307200 environment interactions, zero optimizer updates/replay insertions, unchanged training counters. All prefixes, query counts, source configs/runtime and pre-intervention signals matched. Only3,993,231 bytes of JSON returned locally; checkpoints remained on the server. Seven collection/summary tests passed.

## Findings

The original step-overlap observer does not show an upward change-specific signal: first-post raw surprise exceeds the paired no-change case in only1/160 profile/seed/policy contrasts. Merely querying the old detector once per256 transitions yields tiny first-post reset-probability contrasts, with means between-0.00000221 and0.00001601. Cadence alone is not a demonstrated repair.

Previous-chunk reference gives a clear first-post response under fixed-minus2, with no episode endings in that window. Its no-change mean raw score is5.42 and mean reset probability1.97%; each changed profile has20/20 positive paired raw-score contrasts:

| Target | Mean raw score | Mean reset probability |
| --- | ---: | ---: |
| heavy | 24.74 | 11.92% |
| slippery | 18.03 | 5.90% |
| high_grav | 13.98 | 3.89% |
| compound | 22.72 | 9.71% |

This does not transfer cleanly to the adaptive policy. Its first-post historical-reference raw-score differences versus no-change are-0.239 (heavy),+0.010 (slippery),-0.069 (high_grav),-0.105 (compound). Heavy physically terminates in that window in20/20 probes without producing the intended upward reset response. Its high_grav/compound physical terminations occur in the second post window. Reward-window evidence is strongly policy/trajectory dependent; even the fixed-policy no-change raw score is not negligible. Observer score units differ, so larger scores across observers are not themselves a superiority criterion.

HOLD for RL advancement. Historical reference is a candidate observation, not a validated drop-in belief replacement or a control improvement.

## Next Step

Keep both frozen policies and matched mode interventions. Test a state/action-conditioned transition predictor and standardized prediction residual, using only pre-switch data for fitting/calibration, with an untouched pre-switch validation segment. Compare against the reward-reference candidate and no-change trajectories before another actor-training run. This targets observation quality without reviving NAU/SINDy or expanding training seeds.

## Limits

One trained seed, two existing policies;20 probe seeds are not training replicates and share no-change controls across profiles. Frozen weights, fixed reward scale, extended1536 TimeLimit and differing observer clocks/init histories prevent a direct training or50x benchmark claim. Ending counters combine physical termination and truncation: the first post window precedes TimeLimit; final-window fixed-policy endings cannot be labelled falls from these artifacts. Adaptive trajectories have an earlier physical reset, making TimeLimit impossible in their later post windows. Both post windows remain in the machine-readable results.

Summary: `results/general_ood_v41_frozen_signal/summary.json`.

Follow-up v42 matched replay records termination and truncation separately while exactly reproducing v41 rewards and merged endings. All fixed-minus2 final-window endings were TimeLimit truncations, with no physical failures. This resolves their causes without changing the original v41 artifacts or its HOLD judgment; see `docs/general_ood_v42_frozen_dynamics_results.md`.
