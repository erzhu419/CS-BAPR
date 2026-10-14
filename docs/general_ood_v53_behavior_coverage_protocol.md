# v53: broaden normal behavior coverage

v52's normal-frozen mean coefficient was .01071, whereas normal-updating reached
.91535 under identical normal physics and beta=-2. This study changes only the
reference data's behavior coverage, not the agent, physics or readout settings.

Every trajectory independently loads the v40 fixed-minus2 seed1 final source and
v52's shared fit-only normal replay/context. Collect stochastic actions under
both frozen weights and normal online learning for each fit/calibration seed.
Fit the same quadratic state/executed-action delta predictor on the pooled fit
traces; calibrate the same centered residual score and age-q95 reference on the
pooled, separate calibration traces. All reference physics remain normal.
Do not use these new reference traces to refill test replay.

Preflight: fit98200, cal98210, test98220, each trajectory1200 steps. Formal:
fit98300/1, cal98310/1, test98320--323, each2000 steps. Fit and calibration each
include a frozen and updating branch per seed. Every held-out test seed has five
updating branches: normal plus heavy, slippery, high_grav, compound. Physical
changes occur at step800/1000 for preflight/formal without resetting the state.
All branches use fixed beta=-2, identical initialization/RNG/replay and unchanged
losses. Require exact state/action/transition prefix equality within each pair.

Score every test trace with both the old narrow and new broad reference, frozen
throughout testing. Causal16-step departure history spans episode boundaries;
retain TimeLimit1000. No readout controls actions or updates. Record full/prefix/
first-post50/post/last64 coefficient and signal means, and paired change-minus-
normal contrasts. Learning-normal full and last64 means must be <=.10 (one tenth
of the full actor coefficient range) and all four profiles' paired first50 and
last64 contrasts positive before expanding this coverage diagnostic. This is
not a return-based threshold search or adaptive-penalty RL confirmation.

Preflight costs10800 new interactions and8400 updates; formal costs56000 and
48000 respectively. Existing source training/shared-data costs are inherited,
not included in these added budgets. Save full agent/replay/RNG/context, active
native MuJoCo state, reference traces, completed cells and progress every500
steps and between cells. All large data/checkpoints stay server-side; sync only
scalar JSON/logs. Protect run_shared from staging. Existing inputs reside on
node007; use its scheduler-selected idle GPU with no GPU preference or Windows.

Limitations: one source training seed; held-out continuation RNG seeds measure
conditional detector validity, not independent training-seed return gains.
These four changes are diagnostic physical profiles, not the 50x OOD protocol.
