# v149 matched warm recovery continuation

Status: complete; all four formal tasks exited successfully and their source,
budget and fixed-snapshot evaluation records passed validation. Live
dispatch state is in `experiments/general_ood_v149_warm_recovery_20261014/dispatch_manifest.json`.
v148 shows stable ID policies
already survive2.0x short pulses, while scratch randomization can damage
normal and within-support recovery. Test initialization/training stability
before adding context, symbolic losses, more seeds or harsher environments.

Use the two stable v147 no-pulse600k full states (training seeds5/6). For each,
fork matched no-pulse continuation and randomized-recovery continuation,
100,000 additional native interactions per arm. Restore the same actor,
critics, optimizers, entropy state, normal replay, reset/monitor state and RNGs;
no policy/replay reinitialization or altered learning rates. Keep current
MuJoCo state continuous and assign the new pulse schedule only at subsequent
episode resets. Private pulse RNG is the saved, unused control RNG.

Randomized episodes use the unchanged50% assignment, gravity2.0, duration10,
onsets80--320. No labels/clocks or reward/termination changes. Preserve task
logger and the full dynamic environment resume path. Checkpoint at650k/700k
cumulative steps (50k/100k new), with no best-checkpoint selection.

Evaluate the same v147 nine-condition protocol at both new snapshots, keeping
2.659x and25/50-step shocks outside training support. Report normal stability,
100-step recovery and final survival separately per source seed; these are
two paired continuation seeds, not four new independent training seeds.

Native/local resume smokes precede four freely placed GPU tasks on available
Linux nodes. Checkpoints/replay remain server-side; pull scalar evidence only.
The test is standard-SAC environment calibration, not CS-BAPR contribution.

Results: `docs/general_ood_v149_warm_recovery_results.md` and
`results/general_ood_v149_warm_recovery/summary.json`. Next: frozen final700k
action-noise timing attribution, with same-run no-shock controls; no added
training, checkpoint selection or weakened held-out gravity.
