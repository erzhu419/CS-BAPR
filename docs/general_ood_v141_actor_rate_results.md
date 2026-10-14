# v141 results and next step

All six tasks t147479--t147484 completed exit0:1.8M interactions /1798470
updates. Same physics, rewards, budgets and final reset/noise protocol verified
against the original-rate controls.662083 scalar JSON bytes pulled; full states
remain on Linux servers. Training took97.5--116.9 min.120 final CPU episodes /
32094 steps:100 falls and20 time limits, all20 limits from ID seed4.

| Profile / seed | Original sampled length /15 | v141 length /15 | v141 last5k TD MSE |
|---|---:|---:|---:|
| Friction3 | 74.13 | 136.33 | .756 |
| Friction4 | 148.07 | 146.13 | 4.310 |
| Gravity3 | 91.87 | 93.13 | .302 |
| Gravity4 | 69.07 | 87.40 | .504 |
| ID3 | 1000 | 139.80 | 2.189 |
| ID4 | 1000 | 1000 | 4.459 |

All six logged trajectories are numerically moderate: highest5k mean TD MSE
9.69, final alpha.0206--.0662. Both original explosive seeds become stable,
but all OOD episodes still fall. ID3 regresses despite stable values; its periodic
length grows97.7 at100k ->108.3 at200k ->133.0 at275k ->140.4 final. ID4
retains full survival. This is a numerical repair, not sustained-control success.

Next v142: budget-only full-state continuation of all six v141 models300k ->600k,
300k new interactions/updates each. Keep actor3e-5, critic3e-4 and all other
settings. Test whether ID3 catches up and whether OOD plateaus persist.

## Limitations
Two training seeds; reset/noise repeats are not independent training runs.
Training-rate changes do not prove historical causality.600k continuations
cannot establish superiority over original300k controls. Paper unchanged.
Summary: `results/general_ood_v141_actor_rate/summary.json`.
