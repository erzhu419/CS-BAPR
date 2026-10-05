# Nonlinear load v15: training reset coverage

v14 identified an early state-cost penalty at reset radius 1.0 while the late-phase load benefit remained. Test a single intervention: change RL training resets from radius 0.2 to a repeating 0.2/0.5/1.0 cycle indexed by absolute episode. No additional RNG draws, load changes, loss changes, actuator changes or training-budget changes.

## Frozen Experiment

Fresh training seeds 9/10. Balanced coverage runs all four v11 presets: Jacobian-only centered MLP, no-symbolic centered MLP, BAPR and RE-SAC. Two additional narrow-reset Jacobian-only runs use the same seeds to identify the coverage effect without confounding it with seed changes. Ten formal runs total; no pooling with seeds 7/8.

Every learner trains at load severity 1 for 500 episodes of 200 steps: 100,000 interactions and 99,001 updates. Balanced radius counts are 167/167/166; narrow count is 500 at radius 0.2. Candidate SINDy identification and validation remain at radius 0.2, with the unchanged degree-1 basis, threshold 0.001, 10,000 fitting and 1,000 validation interactions. Reset training RNGs after identification. A reference-builder failure is HOLD, without retuning or fallback.

Use the frozen v14 runtime source (unchanged v11 learner configurations), Python 3.10/3.11 and Torch 2.5.1+cu121 plus the existing pinned numerical packages. Two CUDA preflights exercise candidate/BAPR at seed9 for six episodes, including 201 updates, and all evaluation blocks on two paths. Local tests check paired load draws and exact full-state resume across the 1.0-to-0.2 cycle boundary for all four methods. Formal training saves a full checkpoint every 20 episodes; use only the last checkpoint.

## Evaluation And Decision

Evaluate each reset radius separately at load severities 1/50 on the existing paired paths 60000--60099. Record actor, known local-linear and inverse-actuator control; candidate runs also evaluate their own fitted feedback. Split state and command costs into the first 50 and last 150 steps, and save candidate initial/trajectory action and output-force Jacobian alignment. No updates or fitting during evaluation.

Coverage repair requires, at radius 1.0 and each severity, both seeds to have lower total and early-state costs than their matched narrow-reset candidate. Separately, each radius requires lower candidate mean cost than each learned comparator at both severities on both seeds; own-fitted added value requires the same two-seed wins against own fitted feedback. Full confirmation requires coverage repair and all learner/own-fitted gates at all three radii. A failed gate stays HOLD; do not change thresholds, seeds or checkpoints after inspection.

## Limitations And Operations

Larger radii are now covered training resets, not unseen-state OOD. Severity 50 remains unseen load stress. These are two fresh policies per arm in one observed-load environment, evaluated on reused paths; direct controls are feasible references, not certified optima. v13's failed unseen-reset expansion gate remains unchanged. BAPR/RE-SAC retain the existing shared stabilization defaults.

Scheduler chooses any available compatible Linux GPU, with no preferred node, allowlist or GPU pin and no Windows node. Checkpoints remain remote and outside result directories. Pull only small JSON results or logs. No runtime-efficiency work or manuscript edits.

## Dispatch

Fifteen focused tests passed both locally and in the frozen source tree. CUDA preflights t124897/t124919 passed, including the BAPR full-state checkpoint restore. Initial source-staging and non-symbolic result-save failures are retained in the dispatch manifest; neither changed the scientific protocol. Only about 143 KB of successful preflight JSON responses were pulled, with no checkpoints.

Formal tasks t124924--t124933 all exited 0 on scheduler-selected node007, jtl110gpu and jtl110gpu2, taking 32.1--101.5 minutes. All ten final evaluation JSONs passed the frozen-protocol and paired-control checks. Coverage repair and full confirmation failed; retain HOLD. Source, task specifications, validation and startup observations are saved under experiments/nonlinear_load_v15_coverage_20261002, with results in docs/nonlinear_load_v15_coverage_results.md and results/nonlinear_load_v15_coverage/summary.json. No follow-up tasks have been submitted.
