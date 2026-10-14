# v144 native SAC learnability control

Six fresh runs: seeds 3/4 crossed with ID, friction-only and gravity-only.
Use the frozen v143/v138 environment code and isolated SB3 2.7.0 vendor package.
No edits to shared Python environments or the root CS-BAPR implementation.
Native rewards, observations, health termination, motor limits and physics stay
unchanged: friction multiplier 0.3760603093086394; gravity multiplier
2.6591479484724942; all other factors 1. No profile labels are policy inputs.

Standard unmodified SB3 SAC: two [256,256] ReLU critics, [256,256] actor,
all Adam learning rates 3e-4, gamma .99, tau .005, batch 256, replay capacity
1M, one update/interaction after 255 learning-start interactions. Uniform
actions for those first 255 interactions; Gaussian policy exploration thereafter.
Automatic temperature starts at .2; native target entropy is -3.
Timeouts bootstrap; physical failures do not. Native SB3 collection/training
is used, in 5k-step reporting blocks, with a constant learning-rate schedule.

600k interactions /599745 updates per run; six runs total 3.6M interactions /
3598470 updates. Keep both predeclared 300k and600k policy snapshots, never best.
Deterministic periodic checks every25k use seeds149000--149002 and preserve
training RNG. Save resumable states every25k: model/optimizers, replay, MuJoCo
integration state, native timeout state, reset counter, monitor and all RNGs.

Both policy snapshots evaluate on CPU with resets154000--154004,
deterministic and Gaussian-noise seeds137000--137002, native policy std and1000
steps. Noise uses the existing v126 CPU float32 Generator sequence.20 episodes
per snapshot /240 overall. No replay loads, optimizer updates or context module
in evaluation. Match each budget against v141/v142 respectively.

## Interpretation
This is a standard-learner positive control, not a mechanism-isolating ablation.
Compared with v141/v142, SB3 has two rather than five critics, no context/risk
terms/reward normalization, standard unclipped optimizer gradients, actor LR
3e-4 rather than3e-5, no alpha floor, and255
uniform warmup actions rather than policy actions from step0. Common physics,
seeds, interactions/update counts, target entropy -3 and final noise/reset
budgets are matched. The existing effective beta -2 is a risk coefficient,
not the entropy target.
Success establishes learnability for these tested profiles; failure alone
does not establish physical impossibility.

Three 512-step CUDA smokes cover the three profiles and two predeclared
256/512-step snapshots; 257 updates/run. Only native Linux nodes node007,
jtl110gpu and jtl110gpu2 eligible, no preferences or pins. Publish small JSON
locally; models and replay stay or copy server-to-server.

Implementation: https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/sac.html
