# RandomLoad equilibrium v7: frozen no-NAU factorial

Freeze before training results. V6 read-only centering gives a strong mechanism test but is not trained performance. All four new arms use a tanh-MLP, with no arithmetic heads or arithmetic regularization:

| Arm | Deterministic Mean | Symbolic Bundle |
| --- | --- | --- |
| `csbapr-no-nau-no-target-shift` | `tanh(mu(s))` | critic target penalty 0.01, actor Jacobian weight 0.01 |
| `csbapr-no-nau-no-sindy-no-target-shift` | `tanh(mu(s))` | none |
| `csbapr-centered-mlp-no-target-shift` | `tanh(mu(s)-mu(0))` | same 0.01/0.01 bundle |
| `csbapr-centered-mlp-no-sindy-no-target-shift` | `tanh(mu(s)-mu(0))` | none |

`mu(0)` is recomputed from current weights with gradients through both terms during stochastic action sampling, actor updates and evaluation. The learned log-std head is not centered. The zero observation is a known physical equilibrium in this task, not an asserted property of arbitrary RL environments. No action clipping or reference action anchor is added.

Retain the exact v6 load process, severity-1 training, radius-0.2 resets at every severity, 6D observation, physical matrices/costs, control authority and horizon. Fresh training seeds 0/1/2 for every arm, 500 episodes, 100,000 learner interactions and 99,001 updates, width 64, three critics, batch 256, LR 3e-4 and warmup 1000. Actor decay 1e-4, beta_bc 0, arithmetic regularization 0, target shift 0 and action anchor 0 in all four arms. No coefficient or efficiency search. The non-centered symbolic arm repeats, rather than augments, the v6 no-NAU arm for contemporaneous factorial comparisons.

Symbolic arms still collect 50 identification and five validation episodes (11,000 additional transitions); no-symbolic arms do not. Identification is a separate data cost, so this does not isolate the symbolic bundle from its additional observations. Training is reseeded after identification. Final checkpoint only, full-state episode-boundary resume every 20 episodes. Evaluate all severities 1/2/5/10/20/50 on paired seeds 10000--10019; do not pool episodes as independent trained models.

Primary comparisons: centering with versus without the bundle, removal of the bundle with versus without centering, and their paired training-seed interaction. Report all severities and seed failures. Advance a simplified candidate only with lower mean ID and 50x cost and three matched-seed wins at both; if centered/no-symbolic passes versus centered/symbolic, prefer omitting SINDy. Failure is retained rather than changing weights, environment or the gate. Reuse the existing v6 BAPR/RE-SAC evaluations only as labeled historical baseline comparisons; no new concurrent BAPR/RE-SAC training is included.

Before dispatch: focused actor/Jacobian/checkpoint tests plus disjoint-seed-997 six-episode CUDA preflight. Freeze source and existing offline wheels. Scheduler chooses any idle compatible machine/GPU, without node/GPU preference or restrictions. Only small final evaluation outputs return locally; checkpoints remain on workers. Old v5/v6 negatives, counterfactual labels and manuscript tables are preserved.

## Dispatch

All 80 focused tests passed. CUDA preflight finished six episodes, 1,200 learner steps and 201 updates with exit code 0. The 12 effective tasks are running with confirmed positive optimizer updates: two local and ten on node007, selected without placement preferences. Current IDs and observations are retained in `experiments/random_load_equilibrium_v7_20261001/dispatch_manifest.json` and `results/random_load_equilibrium_v7/startup_observations.json`.

An initial runtime-packaging error was repaired; the failed attempts produced no training data and are retained separately in the manifest. Retries do not add seeds or independent trained models. The frozen algorithm, loss weights and learner budgets were not changed during deployment.

## Completion

All 12 effective tasks finished with exit code 0; final configurations, learner budgets and paired evaluations match the freeze. Both centering comparisons pass the original gate; symbolic-bundle removal fails with and without centering. The accuracy-first candidate is centered/no-NAU with the bundle retained. Results and the independent-validation plan are in `docs/random_load_equilibrium_v7_results.md`; machine-readable evidence is in `results/random_load_equilibrium_v7/summary.json` and `completion.json`. Historical negatives and manuscript tables remain unchanged.
