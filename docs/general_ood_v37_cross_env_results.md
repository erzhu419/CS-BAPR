# v37 partial results (2026-10-05)

16/24 cells have exit-0 completion and valid compact evaluation JSON.
Pendulum is complete; Hopper lacks no-context seeds 1/2 and all BAPR/RE-SAC
seeds. The missing cells failed before training on local without a compatible
CUDA runtime. Runtime-recovery tasks are `t135519` through `t135526`;
`t135523` and `t135525` had another incompatible Python runtime at startup
and are replaced by `t135528` and `t135531`. The eight missing cells are
now assigned to node007 and jtl110gpu2, with the original commands unchanged.

| Pendulum method | ID return | Markov 50x return |
|---|---:|---:|
| stable-context candidate | -909.15 | -1121.06 |
| no-context control | -616.01 | -965.57 |
| BAPR stable-context | -913.81 | -1097.42 |
| RE-SAC | -478.35 | -956.46 |

Candidate-minus-RE-SAC OOD gaps by training seed are -80.02, -270.07 and
-143.70 (mean -164.60; descriptive 95% paired t interval -404.89 to 75.70).
This is a consistent three-seed loss, not evidence of bus-like superiority.
Hopper candidate OOD returns are 821.41, 250.91 and 333.12; its missing
baselines prevent a cross-environment verdict.

## Limitations And Next Step

All presets disable symbolic/Jacobian losses. The no-context control also
changes BC and BAPR penalty weights, so this is not a pure context ablation.
Hopper's 2000 episodes produce policy-dependent interaction budgets (completed
runs: 1.69-1.82 million); no separate Hopper cost is recorded.
Complete the unchanged Hopper matrix first. Then judge whether fixed-interaction
and single-factor controls are justified; do not tune v37 or rewrite the paper's
claims from this partial result. Seed values, fixed profiles and intervals are
in `results/general_ood_v37_cross_env/summary.json`.
