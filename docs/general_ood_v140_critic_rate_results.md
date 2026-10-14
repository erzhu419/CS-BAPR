# v140 results and next step

All six tasks t147080--t147085 completed exit0. Each trained300k steps /
299745 updates; total1.8M steps /1798470 updates. Matching physics, native
rewards, configurations except critic lr, and evaluation resets/noise verified.
120 CPU episodes /10081 steps; all physically terminate.662391 JSON bytes
pulled locally, snapshots/replay remain on servers. Training took86.7--130.5 min.

| Profile / seed | Original sampled length /15 | Reduced critic lr length /15 | New last5k TD MSE |
|---|---:|---:|---:|
| Friction3 | 74.13 | 55.87 | 4.47e10 |
| Friction4 | 148.07 | 16.47 | 2.18e7 |
| Gravity3 | 91.87 | 87.07 | .895 |
| Gravity4 | 69.07 | 82.67 | .727 |
| ID3 | 1000 | 81.47 | 1.37e10 |
| ID4 | 1000 | 177.73 | 5.61 |

The smaller critic step resolves gravity4's numerical explosion but does not
repair friction3. Previously stable friction4 and ID3 now diverge; ID4 stays
numerically moderate but loses sustained locomotion. Reject this as the working
configuration. At50k, friction3/4 TD MSE is145/156 with alpha.185/.182; ID3
TD error grows from29 at50k to219 at70k while alpha stays.15. Extreme alpha
growth remains later than TD growth.

Next v141 restores original critic lr3e-4 and changes only actor lr1e-4 ->3e-5
against the original v137/v138 controls: same profiles, seeds3/4, budget,
physics and evaluation. The relative-update-rate hypothesis is not established
by v140, which changed both absolute critic speed and the critic/actor rate ratio.
See `docs/general_ood_v141_actor_rate_protocol.md` and its dispatch manifest.

## Limitations
Two training seeds, not fifteen independent seeds. Rates alter trajectories;
this is not a historical root-cause proof or a general claim about learning
rates. Stable short OOD survival remains a separate problem. Paper unchanged.
Summary: `results/general_ood_v140_critic_rate/summary.json`.
