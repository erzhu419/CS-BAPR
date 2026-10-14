# v132 results and next step

All four tasks t143265--t143268 finished with exit0 and synced their small
source JSONs. Each received100000 interactions /99745 updates; total400000 /
398980. Native training/evaluation took26.8--32.0 minutes per task on the two
new Linux nodes. Checkpoints/replay remain on servers; old results are unchanged.

| Seed | TD loss | Final return | Velocity | Length | Falls /5 | Last5k mean TD MSE |
|---|---|---:|---:|---:|---:|---:|
| 3 | MSE | 1977.05 | 2.1845 | 613.6 | 5 | 5.3463 |
| 3 | 2xHuber10 | 407.59 | 1.9330 | 139.0 | 5 | 1.7161 |
| 4 | MSE | 485.19 | 1.9414 | 165.0 | 5 | 4.8568 |
| 4 | 2xHuber10 | 802.64 | 1.9435 | 268.2 | 5 | 4.9499 |

Huber reduces seed3's final-block squared TD error but harms survival/return;
it improves seed4 survival/return without reducing final-block squared error.
There is no consistent Huber winner. All20 final episodes physically terminate:
forward motion is acquired, sustained healthy locomotion is not.

## Next Step

Propose v133: continue all four final100k full states to300k, preserving replay,
optimizer/alpha/target/context/RNG and environment state. Add200k interactions
and updates per arm (800k total), change only budget, keep both losses and fixed
delta10. No threshold sweep, additional seeds, checkpoint selection, retention
intervention or OOD evaluation yet. Planned, not dispatched.

## Limitations

All four training runs had zero time-limit truncations, so mask correction was
inactive in this pilot and cannot explain better historical seed3 stability.
Old and new MSE configurations match, but training GPUs changed2080Ti to3080Ti
and early trajectories differ; GPU/numerical effects are plausible, not isolated.
Five reset episodes per model are not five training seeds; only two seed pairs
exist. Logged block metrics are update means, including per-batch maxima, not
global peak certificates. Results concern normal Hopper, not OOD superiority.
