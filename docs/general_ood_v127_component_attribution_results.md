# v127 results and next step

Tasks t142369--t142371 all exited zero and automatically synced their JSON
(5145788 bytes); no model/replay download. Batch wall time: 3.73 minutes.
Validated 780 episodes / 379831 evaluation interactions, all 210 coherent v126
controls reproduced, zero training/optimizer updates. Summary:
`results/general_ood_v127_component_attribution/summary.json`.

Full sampled factorial below: component ages in k steps; each seed entry gives
legacy/fresh mean length and falls out of 15 episodes per noise group.

| Mean / std / context | Seed0 length; falls | Seed1 length; falls | Seed2 length; falls |
|---|---|---|---|
| 300 / 300 / 300 | 146.1 / 145.7; 15 / 15 | 719.5 / 871.6; 7 / 8 | 1000 / 1000; 0 / 0 |
| 300 / 300 / 350 | 145.8 / 145.4; 15 / 15 | 696.3 / 833.6; 9 / 8 | 1000 / 1000; 0 / 0 |
| 300 / 350 / 300 | 146.1 / 145.9; 15 / 15 | 620.0 / 748.1; 12 / 12 | 1000 / 1000; 0 / 0 |
| 300 / 350 / 350 | 145.9 / 145.7; 15 / 15 | 650.3 / 806.3; 11 / 9 | 1000 / 1000; 0 / 0 |
| 350 / 300 / 300 | 149.2 / 150.1; 15 / 15 | 1000 / 702.8; 0 / 7 | 109.9 / 107.9; 15 / 15 |
| 350 / 300 / 350 | 149.7 / 150.5; 15 / 15 | 845.9 / 464.1; 3 / 10 | 109.8 / 107.6; 15 / 15 |
| 350 / 350 / 300 | 148.8 / 149.4; 15 / 15 | 924.9 / 486.4; 2 / 10 | 110.1 / 107.8; 15 / 15 |
| 350 / 350 / 350 | 149.3 / 149.8; 15 / 15 | 759.1 / 397.5; 4 / 10 | 110.1 / 107.5; 15 / 15 |

Seed2: all 120 episodes using the complete 300k mean function reached the time
limit; all 120 using the 350k mean fell. Swapping only the mean into the coherent
350k policy changed length from 110.1/107.5 to 1000/1000 and return from
270.3/260.4 to 3292.8/3290.7. Neither old std nor old initial normalizer rescued
the 350k mean. All four deterministic mean/context combinations reached 1000.
The loss of robustness is attributable to the mean function in this frozen
factorial, rather than std-function or initial-normalizer replacement.

Seed1: deterministic old-mean lengths were 440.8/654.0 with old/new context;
both new-mean combinations reached 1000. New mean with old std and old context
gave sampled length 1000/702.8 and falls 0/7, versus coherent 350k 759.1/397.5
and 4/10. Mean improvement and std/context interactions coexist. Seed0 fell in
all sampled and deterministic combinations; its mean-policy failure persists.

## Next Step

v128 submitted after both native smokes passed; see
`experiments/general_ood_v128_mean_retention_20261011/dispatch_manifest.json`.
Fork each seed's full v123 300k training snapshot into
two 50k continuations, unchanged control versus mean-only source regularization.
Keep beta=-2, weight_reg=0, native reward/physics, replay and optimizer resume.
Freeze the full 300k reference actor. On the same current augmented replay
states, add `alpha.detach() * D_mean`, where
`D_mean = 0.5 * mean_batch(sum_action(((mu - mu_ref) / std_ref)^2))` in pre-tanh
coordinates. This is a mean-shift penalty normalized by frozen reference std,
not a constraint on the current std function or an old context normalizer.
Use coefficient one for every seed, without a per-seed choice or parameter sweep.

Six paths / 300000 new training interactions; final 350k snapshots only. CPU
evaluation uses five common resets, deterministic actions and full-scale noise:
legacy 12400--12402, v126 12600--12602, and new heldout 12800--12802, separately
reported (300 final episodes). The decision is whether seed2 is rescued without
losing seed1's improvement or worsening seed0; retain all outcomes. Add no OOD
profiles or training seeds until this normal-environment foundation is sound.

## Limitations

Three independent training seeds; reset/noise repeats are not additional seeds.
Both existing noise groups were already seen in v126. Crossed actors are frozen
diagnostics, not adopted policies. Attribution does not identify the particular
training-loss term responsible for mean drift or show that regularization will
repair it. Initial-normalizer swaps do not test every possible context change.
The 300k seed0/1 references are imperfect, so tethering them may impede learning.
These are normal Hopper results, not evidence of CS-BAPR OOD superiority.
Paper tables and earlier negative results remain unchanged.
