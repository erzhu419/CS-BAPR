# v55: match age calibration to online learning

v54 passes learning-normal bounds and all8 tails; gravity seed98420 ties normal
in first50 despite a higher residual signal. Test whether pooling frozen-policy
and learning-policy threshold populations obscures this early shift. Preserve
the old HOLD; this is a new calibration experiment, not a revised v54 verdict.

Freeze v53 broad dynamics and v54 signed-window center, covariance, precision,
ridge and global score-q95. Only refit the age-q95 reference using normal online
learning. Keep all11 coordinates, residual window16 and departure history16;
neither history resets at episode boundaries. The old pooled reference stays
frozen as the paired readout control. Do not use an updating flag at inference.

Preflight first reproduces v54 calibration seed98410, frozen/updating1200 steps
each; inspect q95/max by producer at ages800/816/832/848. These audit traces never
fit the new reference. New calibration98600/1 uses1200 normal interactions/updates
per seed. Untouched tests98620/1 each have normal plus four changed profiles.
Formal, only after a pass, uses calibration98700/1 and tests98720--323; no audit
repetition. Every trajectory starts from v40 fixed-minus2 seed1 and v52 fit-only
replay/context. Stochastic actions, fixed beta=-2, TimeLimit1000, step800 change,
exact paired prefixes/switch inputs, unchanged losses. Test traces never refit
reference data or augment another trajectory's replay.

Retain the existing criteria: every normal full/last64 mean coefficient<=.10;
every changed profile/seed first50/last64 paired contrast>0. No return or OOD
selection of thresholds. Added preflight cost16800 interactions/15600 updates;
formal26400/26400. Existing model/reference collection costs remain inherited.
Save full progress, replay/RNG/context and native state every500 and per-cell.
Only scalar JSON/logs return; all raw data/checkpoints remain server-side.

Inputs reside on node007; scheduler chooses idle GPU with no preference or
Windows. Scope: an online-learning calibration, not a frozen-policy deployment
detector. One source seed and fixed beta; no adaptive RL or50x OOD return claim.
