# RandomLoad v6: read-only centering results

All nine checkpoint diagnoses completed on their existing workers. Original per-episode returns reproduce within maximum absolute error 0.000021. Three focused checks passed: pre-tanh centering/bounds, Jacobian input partition and paired reference rollout. Only 111,319 bytes of remote JSON were downloaded; no optimizer updates or checkpoint transfers occurred.

Mean cost over three training seeds; counterfactual actions are `tanh(mu(s)-mu(0))`.

| Existing Actor | Original ID | Centered ID | Original 50x | Centered 50x |
| --- | ---: | ---: | ---: | ---: |
| Full CS-BAPR | 46.337 | 0.142 | 48.280 | 1.292 |
| No NAU: tanh-MLP + symbolic bundle | 16.436 | 0.143 | 18.246 | 1.243 |
| No SINDy: arithmetic actor | 41.403 | 24.110 | 45.056 | 29.005 |

Every checkpoint improves at ID and 50x after centering. The two symbolic arms become close to their saved fitted-feedback cost (ID 0.140, 50x 1.197); this intervention shows that constant mean offset accounts for most of their excess evaluation cost on these paths. No-SINDy retains a large failure: seed 0's centered 50x cost is 74.069, versus 3.452/9.493 for seeds 1/2. Centering alone does not fix that arm's state-dependent policy. Removing NAU remains the simpler candidate even after centering.

## Limitations

These are post-hoc counterfactuals on the same evaluation paths, not new trained models or confirmation of training-time centering. No-SINDy still uses the arithmetic actor; there is no matched tanh-MLP/no-symbolic arm yet. Thus this diagnosis does not isolate SINDy from architecture or the two symbolic losses. It inherits v6's controlled linear environment, extra symbolic identification data and heterogeneous Torch runtimes. Prior results and manuscript tables remain unchanged.

## Next Step

Train a frozen 2x2 comparison with no NAU in any arm: ordinary versus equilibrium-preserving tanh-MLP, each with versus without the symbolic bundle. Keep the v6 environment, weights, learner budgets and seeds 0/1/2 fixed. This separates the effect of enforcing the known physical equilibrium during learning from the benefit of the symbolic bundle; it also directly tests whether both NAU and SINDy can be omitted. Compare against retained v6 BAPR/RE-SAC evaluations as explicitly reused references, not fresh simultaneous controls.

Evidence: `results/random_load_v6_centering/summary.json` and nine per-checkpoint JSONs; reproduce with `python3 scripts/summarize_random_load_centering.py`.
