# v147 randomized recovery-training plan

Status: complete; results are in `docs/general_ood_v147_randomized_recovery_results.md`.
Dispatch state is recorded in
`experiments/general_ood_v147_randomized_recovery_20261014/dispatch_manifest.json`.
v145/v146 show both a learning threshold and
delayed failure after a brief extreme pulse. Test recovery-state coverage before
another learning-rate/budget extension or mechanism attribution.

Four fresh standard-SAC runs: training seeds5/6 crossed with no-pulse control
and randomized pulse/recovery training. Same native Hopper rewards, motor
limits, healthy termination, SAC settings and600k interaction budgets; retain
predeclared300k/600k snapshots. Use the task-local persistent logger fix.

Training starts at normal gravity1.0. In the randomized arm, half the episodes
receive one2.0-gravity pulse lasting10 steps, with onset uniformly sampled after
80--320 completed steps, then return to1.0 without resetting state. Pulse RNG
is independent of policy sampling and preserved in full-state checkpoints.
Neither mode labels nor onset clocks are policy inputs.

Evaluate both arms with the unchanged v146 nine-condition protocol, matched
resets/noises and1000-step cap. Gravity2.659 amplitude and25/50-step durations
remain outside training support; keep no-pulse and permanent controls. Report
100-step recovery AND eventual full survival, per training seed.

If randomized training improves unseen-pulse survival, use this calibrated
environment for matched CS-BAPR/BAPR/RE-SAC mechanism comparisons. If it does
not, retain the negative result and diagnose recovery-state support. Training
randomization alone must not be relabeled as a CS-BAPR contribution.
