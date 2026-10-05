# RandomLoad v6: frozen read-only centering diagnosis

Freeze before reading counterfactual results. Use the nine existing final checkpoints: full CS-BAPR, no-NAU and no-SINDy, training seeds 0/1/2. Load them on their existing workers; no checkpoint transfers, parameter changes, optimizer updates or new training. Use the frozen v6 environment and source.

Compare original deterministic actions with the post-hoc counterfactual `tanh(mu(s)-mu(0))`, where `mu(0)` is the saved actor's pre-tanh zero-observation mean. Evaluate 1x/20x/50x with paired seeds 10000--10019 and 200-step horizons. Reproduce each original per-episode return before accepting the counterfactual analysis; CPU/batched replay allows relative tolerance 1e-4 and absolute tolerance 1e-3. Failure stops scientific interpretation of that run.

Record trajectory cost, saturation, physical-state RMS, last-50-step physical-state RMS and final position/velocity norms. At every tenth pre-action step, measure action RMSE and squared-Frobenius Jacobian errors separately for physical-state and force-input columns, against the saved fitted feedback for symbolic arms or a known-model diagnostic for no-SINDy. Evaluate that reference separately. Return only small JSON files.

A consistent ID and 50x improvement across the three seeds would support testing equilibrium preservation during training; a failure keeps this centering hypothesis negative and redirects attention to state-dependent policy errors. These are counterfactual diagnostic policies, not trained algorithms or confirmatory evaluations. Do not use this test to choose an environment or loss coefficient. The user's permission to drop NAU or SINDy permits a simpler next candidate; matched tanh-MLP controls are still required to attribute its gains.
