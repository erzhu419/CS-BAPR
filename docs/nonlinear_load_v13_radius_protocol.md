# Nonlinear load v13: reset-radius stress test

Reuse v11's frozen seed7/8 policies and fitted references through v12's existing inference JSONs. Evaluate Jacobian-only, no-symbolic, BAPR and RE-SAC at physical-state reset radii 0.2/0.5/1.0, load severities 1/50 and paired paths 60000--60099. No training, refitting, loss changes, new checkpoint selection or disturbance changes. This is a stress test on the already-used v12 paths, not additional held-out paths or training seeds.

Set the existing environment's `amplitude` before reset. Keep the original ball-uniform reset draws and identical action-independent load RNG streams across radii and controllers. Do not modify environment dynamics or global constants. Record the actual radius in each radius-specific environment specification; keep source training settings separate.

Each task first evaluates radius 0.2 and reproduces every v12 actor/control return at ID and 50x within the established rtol=1e-4, atol=1e-3. A reproduction failure stops the task before expanded radii. Then evaluate 0.5 and 1.0 without changing policy parameters or gains. Save raw paired returns, per-path maximum physical-state norm, action saturation and observation-unit-ball exit fractions, plus configuration, runtime and reproduction evidence.

Apply the previous two-seed ID/50x comparison and own-fitted-feedback gates separately at each expanded radius. Both radii must pass to call the expansion confirmed; one successful radius cannot offset another's failure. Preserve v11/v12 gates and all negative paths. Unit-ball observation fractions diagnose neighborhood coverage and are not an additional threshold to tune.

Eight formal tasks contain 15,600 episodes including controls. Two preflights cover centered-tanh and ReLU actors at all three radii on path 60000 (42 episodes including controls). All new training interactions, identification interactions and optimizer updates are zero. Reuse the pinned v12 runtime, keep full checkpoints on workers, and return only small JSONs. Scheduler selects idle compatible Linux/WSL resources without node/GPU preferences or allowlists; no Windows dispatch. Manuscript tables remain unchanged.

## Dispatch

Both radius/path unit tests passed. CUDA preflights t121899/t121900 exited 0, completed all three radii and reproduced radius 0.2; their maximum absolute return deviations were 7.24e-7 and 6.49e-5. Formal seed7 tasks are t121904/t121905/t121907/t121908; seed8 tasks are t121909/t121910/t121911/t121912, in Jacobian-only/no-symbolic/BAPR/RE-SAC order. Snapshot, inference inputs, source evaluations and task records are in `experiments/nonlinear_load_v13_radius_20261002/`; task submission adds no node or GPU constraints.

All eight formal tasks exited 0 and passed complete radius-0.2 reproduction. Each expanded radius passes the learner-comparison gates but fails the own-fitted added-value gate, so full expansion confirmation is HOLD without changing any criterion. Results and paired failure paths are in `results/nonlinear_load_v13_radius/summary.json`; interpretation and the next read-only diagnostic are in `docs/nonlinear_load_v13_radius_results.md`. No follow-up tasks were submitted.
