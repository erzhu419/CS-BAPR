# v136 results and next step

Tasks t144321/t144322 completed exit0 on node007 CPU, with all three Linux
nodes eligible and no preferences.400 episodes /147771 evaluation steps,
zero training/optimizer updates. All200 paired prefixes match exactly through
step201; every episode reaches the intervention.21 tests passed. Only1363263
bytes of scalar JSON pulled, including smokes; no checkpoints or trajectories.

Mean post-warmup survival, sampled mode (15 runs per cell; online / held):

| Condition | Seed3 | Seed4 |
|---|---:|---:|
| ID | 800.00 /664.80 | 800.00 /140.33 |
| Easy | 131.07 /109.33 | 57.07 /58.07 |
| Heavy | 71.73 /70.73 | 70.27 /63.67 |
| Mega | 38.33 /37.87 | 39.13 /38.47 |
| Slippery | 77.07 /76.27 | 80.67 /71.00 |

All160 online OOD episodes fall, as do all160 held-context OOD episodes.
Easy failures violate the angle bound; heavy/mega/slippery failures violate
the height bound in both arms. ID online survives all40 episodes. The large
ID hold penalty and small/inconsistent OOD improvements do not establish
environment adaptation; short-transition context also supports ordinary gait.

Next v137, planned not dispatched: matched fixed-profile learnability probes
at unchanged intensity50. Fresh ID/easy/mega MSE policies, seeds3/4, the same
frozen training driver/config and300k steps each: six runs /1.8M interactions.
Evaluate predeclared final snapshots in their matching static profiles using
five resets154000--154004, deterministic plus noises137000--137002. If the
OOD-specific policies learn, proceed to matched regime-randomized training;
otherwise investigate task/training suitability before interpreting adaptation.

## Limitations
Two training seeds; inference intervention is not a training ablation. v136
does not establish physical infeasibility or RE-SAC/BAPR superiority. Summary:
`results/general_ood_v136_warmup_context/summary.json`. Paper tables unchanged.
