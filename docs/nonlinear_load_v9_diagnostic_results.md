# Nonlinear load v9: read-only diagnosis

Both Linux CPU diagnostics completed (seed5/node007, seed6/jtl110gpu): 2,400 episodes covering four controls, severities 1/20/50 and all 100 paired paths. Nine focused tests passed, including finite-difference output-force Jacobians, command/state cost decomposition and read-only checkpoint loading. All saved returns reproduced; maximum absolute error is 1.05e-5. Pulled only two JSONs totaling 340,180 bytes, with zero checkpoint transfers, refits or optimizer updates.

Mean 50x cost decomposition across training seeds 5/6:

| Control | State cost | Command cost | Total cost |
|---|---:|---:|---:|
| Actor | 1.224308 | 2.567089 | 3.791397 |
| Own fitted | 1.181705 | 2.597996 | 3.779701 |
| Known local-linear | 0.222493 | 2.636611 | 2.859103 |
| Known inverse-actuator | 0.155569 | 2.570781 | 2.726351 |

The actor's 0.932294 excess versus known local-linear decomposes into +1.001816 state cost and -0.069521 command cost. Against known inverse feedback it is +1.068739 state cost and -0.003692 command cost. There is no actor action saturation. Thus the remaining observed gap is state regulation, not excessive command effort or exhausted authority.

Relative to its own fitted reference, seed5 adds 0.084171 state cost while saving 0.043259 command cost, leaving a 0.040912 loss. Its worst paths 50002/50044 add 1.674310/1.553495 state cost, explaining almost all their 1.697689/1.534996 total excesses. Seed6 also adds 0.001036 state cost, but saves 0.018554 command cost and wins overall by 0.017518. The two-seed mean remains worse than fitted feedback by 0.011697; the failed added-RL-value gate stands.

At the origin, the physical output-force derivative with respect to current load is `[-0.342, +0.054; -0.597, -0.415]` for seed5's actor, versus `[-0.352, +0.052; -0.596, -0.416]` for its fitted reference. Seed6's actor is `[-1.052, +0.061; -0.390, -0.470]`, versus `[-1.050, +0.064; -0.390, -0.472]` fitted. Both retain under-compensation and cross-axis coupling, unlike the known controllers' `-I` derivative at the origin. On 50x actor trajectories, actor-versus-fitted load-action-Jacobian squared errors are only 4.15e-5/2.30e-5; action-component RMSEs are 0.009882/0.005128. The close reference tracking extends beyond resets.

## Next Step

Run only the missing symbolic 2x2 ablation cells, seeds 5/6: critic-only (`weight_sym=0.01`, `jac_weight=0`) and Jacobian-only (`weight_sym=0`, `jac_weight=0.01`). Reuse frozen v9 full/none arms as controls; keep the same environment, fitting budget, backbone, training budget and paired evaluation protocol. Inspect both performance and whether load-feedback bias persists. Keep ID/50x matched-seed and added-RL-value criteria unchanged. If neither cell improves on its own fitted controller, prioritize reference identification rather than more seeds or another severity change. No new training was submitted by this diagnostic pass.

## Limitations

Two trained policies and descriptive diagnostics do not identify which loss term causes reference tracking; the proposed ablation is needed. Known controllers are privileged feasible references, not optima, and instantaneous cancellation is not an optimality condition under command cost. No replay-gradient attribution was run. V9 training negatives and manuscript tables remain unchanged.

Reproduce aggregates with `python3 scripts/summarize_nonlinear_load_v9_diagnostic.py`; worker commands and input locations are in `experiments/nonlinear_load_v9_diagnostic_20261002/manifest.json`. Per-path costs and trajectory metrics are in `results/nonlinear_load_v9_diagnostic/seed5.json` and `seed6.json`.
