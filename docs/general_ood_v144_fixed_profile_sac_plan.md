# v144 fixed-profile SAC positive-control plan

Status: six formal tasks t148407--t148412 completed exit0 after nine local tests and
three native CUDA smokes. v143 confirms posture/control failure, but not
whether a standard learner can solve these fixed physics. Test that question
before changing rewards, healthy termination, motor limits or OOD profiles.

Use an established standard-SAC implementation, not another hand-written
variant of the current risk/context learner. Six fresh runs: seeds3/4 crossed
with ID, friction-only and gravity-only. Preserve v138 native observations,
actions, rewards, timeouts and exact profile multipliers. No profile labels,
context module, symbolic fitting or CS-BAPR constraints in this positive control.

Budget600k interactions/run; retain both predeclared300k and600k snapshots.
Evaluate both on CPU with resets154000--154004, deterministic plus native-scale
noises137000--137002,1000-step cap. Compare300k against v141 and600k against v142;
do not select a best checkpoint or pool reset/noise repeats as training seeds.
Use matched common SAC settings and document implementation-specific differences.
Implementation and exact differences: `general_ood_v144_fixed_profile_sac_protocol.md`.

If SAC solves a changed profile, keep its physics and investigate the present
learner. If it also fails, calibrate profile learnability with milder physical
interventions before returning to hidden-regime adaptation. Neither outcome
alone is a proof of physical impossibility or CS-BAPR superiority.

Native implementation acceptance is required before release. Only node007,
jtl110gpu and jtl110gpu2 eligible, no preferences/pins/Windows. Logs/scalar JSON
only locally; all checkpoints/replay remain on servers.
