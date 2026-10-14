# v123 results and next step

All three tasks t142036/t142037/t142038 exited zero in 86.7--87.8 minutes.
Each unchanged v122 source continued from 100000 to 300000 total interactions
and 299745 total optimizer updates. New formal interactions/updates: 600000;
reused source interactions: 300000. Only 199982 bytes of scalar JSON were synced.

| Seed | Velocity at 100k -> 300k (m/s) | Length at 100k -> 300k | Return at 300k | Falls / 5 at 300k |
|---|---:|---:|---:|---:|
| 0 | 1.265 -> 1.826 | 102.2 -> 147.0 | 415.372 | 5 |
| 1 | 2.255 -> 2.505 | 249.2 -> 440.8 | 1544.823 | 5 |
| 2 | 1.707 -> 2.301 | 1000.0 -> 1000.0 | 3300.371 | 0 |

Mean velocity/length/return improved by 0.468614 m/s, 78.8 steps and 504.030.
All seeds improved return, but final sustained locomotion remains 1/3 seeds and
10/15 final episodes end in falls. Seed 0 plateaus near 147 steps. Seed 1 survived
all three periodic evaluation episodes at 275k, then fell in all five final
episodes at 300k (lengths 432, 426, 414, 449, 483). Seed 2 also alternated between
long and short periodic episodes before its successful final snapshot.

Frozen mean Q remains 201.196/229.578/265.160, with zero target L1 uplift and
successful loss reconstruction. The earlier Q-scale inflation has not returned.
Final 5k-block critic losses are 1.724/12.808/36.559. A small ensemble disagreement
value alone does not establish that its actor gradient is harmless.

## Next step

Plan v124 as a read-only normal-environment diagnosis on all three final sources:
compare deterministic versus sampled actions on common reset seeds, record the
physical condition causing each fall, and decompose Q, uncertainty, entropy and
behavior-cloning actor gradients on frozen replay pools. Preserve the context
and random state, perform zero optimizer updates, and keep all replay/checkpoints
native. This distinguishes action-extraction sensitivity from persistent gait
failure and identifies which loss term merits a single-factor training test.
No new tasks have been submitted for this plan.

## Limitations

Three training seeds and five final episodes per seed do not establish broad
source reliability or OOD superiority. The 300k-versus-100k comparison changes
training budget; it is not a matched-budget L1 ablation. Periodic results diagnose
instability but do not select a replacement checkpoint. No source readout refit
or manuscript-table change was made.

Summary: `results/general_ood_v123_source_extension/summary.json`.
