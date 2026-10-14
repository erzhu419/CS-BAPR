# v128 results and next step

All six tasks t142411--t142416 exited zero and automatically synced small JSON
(2760161 bytes); no checkpoint/replay download. Batch wall time: 34.27 minutes.
Validated six matched paths / three training seeds / 300000 new interactions and
updates / 300 final episodes / 154461 evaluation interactions. All 105 old CPU
control episodes reproduced v126's coherent 350k models. Summary:
`results/general_ood_v128_mean_retention/summary.json`.

Entries below are mean episode length / falls. Deterministic groups have five
episodes; each sampled group has 15. All sampling uses full noise scale one.

| Seed | Arm | Deterministic | Legacy | v126-seen | New heldout |
|---|---|---|---|---|---|
| 0 | Control | 148.8 / 5 | 149.3 / 15 | 149.8 / 15 | 150.1 / 15 |
| 0 | Mean retention | 149.2 / 5 | 148.1 / 15 | 147.8 / 15 | 148.6 / 15 |
| 1 | Control | 1000 / 0 | 759.1 / 4 | 397.5 / 10 | 1000 / 0 |
| 1 | Mean retention | 1000 / 0 | 1000 / 0 | 1000 / 0 | 959.2 / 2 |
| 2 | Control | 1000 / 0 | 110.1 / 15 | 107.5 / 15 | 109.6 / 15 |
| 2 | Mean retention | 1000 / 0 | 1000 / 0 | 528.1 / 8 | 1000 / 0 |

Seed2's sampled falls decreased from 45/45 to 8/45; heldout return increased
from 267.9 to 3286.3. Its residual angle falls were three resets with noise12600
and all five with noise12601, at steps 114--117. Seed1's sampled falls decreased
from 14/45 to 2/45, retaining deterministic improvement, but the heldout group
worsened: control 1000/0 versus retention 959.2/2, return 3260.1 versus 3181.1.
The new falls used noise12801 at steps 451 (height) and 937 (height and angle).
Seed0 still fell in every episode; its heldout return slightly declined from
431.4 to 422.3. Mean retention preserves a useful policy; it did not repair this
already-bad reference.

Last-5k replay-batch mean-shift distances (control/retention) were 40.825/1.077,
1.634/0.464 and 1.617/0.491 for seeds 0/1/2. Retention penalties stayed positive
and finite. These batches come from different visited states; they are training
diagnostics, not matched-input evidence of the remaining failure's cause.

## Next Step

v129 submitted after native smoke passed; see
`experiments/general_ood_v129_heldout_robustness_20261011/dispatch_manifest.json`.
Freeze the v123 final 300k reference and both v128
final 350k arms for every training seed. Evaluate five common resets and 20 new
noise seeds 13000--13019 at scale one, plus five deterministic episodes/model.
Three CPU tasks / nine models / 945 episodes / zero training updates, native
1000-step limit. Reproduce all 45 deterministic baselines; report per-training-
seed results and noise-wise paired changes. This broadens heldout noise coverage
without another training sweep or checkpoint selection. Keep the reference as
context for the comparison, not as a best-checkpoint candidate. Decide whether
the benefit persists beyond three-noise groups before changing the penalty or
adding OOD conditions. Checkpoints stay native; sync small scalar JSON only.

## Limitations

Three independent training seeds; repeated resets/noise are not additional
training seeds. Seed1 worsened on the first heldout group, seed2 retains failures,
and seed0 is unrepaired. The penalty indirectly changes other learned quantities
through training; this is not proof that its effect bypasses std/context. These
are normal Hopper results, not CS-BAPR OOD superiority or submission readiness.
Paper tables and earlier negative results remain unchanged.
