# v57: separate initial and post-reset age calibration

Freeze v53 dynamics and v55 formal signed-window center/covariance/precision,
ridge and global score95. Collect normal online-learning calibration only.
Each cell runs2000 steps: two complete TimeLimit1000 episodes, requiring both
calibration populations to cover ages0--999. Use two age-q95 curves, selected
only by whether an actual prior episode boundary has occurred. Do not use an
OOD label or reset the residual/departure histories. Settings remain16/8/.95/16.

Three readouts share each test trajectory: frozen v55 formal control,
matched-pooled age calibration, and initial/post-reset candidate. The last two
use exactly the same normal data, budget and score scale; the pooled control
separates the phase intervention from the longer calibration budget.

Preflight calibration98800/1, tests98820/1; formal calibration98900/1, tests
98920--98923, only after preflight passes. Every test is1200 updates/interactions,
normal or one of four original profiles, step800 change and exact paired prefix.
Added preflight16000 interactions/updates; formal28000/28000. Source remains
v40 fixed-minus2 seed1 plus v52 fit-only replay/context, stochastic actions,
unchanged losses and beta=-2. Readouts never control learning or actions.

Require every candidate normal full/last64 mean<=.10 and every changed pair's
first50/last64 contrast>0. No threshold selection using test/OOD outcomes.
Save native progress/RNG/replay every500 steps and between cells. Only scalar
JSON/logs return. This is an initial-versus-later episode detector diagnostic,
not a later-lifetime invariance or adaptive RL/50x OOD performance claim.
