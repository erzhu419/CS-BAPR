# v125 results and next step

All six tasks t142173--t142178 exited zero and automatically synced scalar JSON
(1002824 bytes); batch wall time was 32.12 minutes. Paired protocol validation
passed. Each arm added 50000 interactions/updates to the same 300k full source,
ending at 350k interactions / 349745 updates. Six paths added 300000 training
interactions. Final CPU evaluation covered 120 episodes / 43244 interactions.
Checkpoint selection remained final-only; checkpoints and replay stayed native.

| Seed | Actor beta | Deterministic length (falls / 5) | Deterministic return | Sampled length (falls / 15) | Sampled return |
|---|---:|---:|---:|---:|---:|
| 0 | -2 | 148.8 (5) | 429.5 | 149.3 (15) | 430.2 |
| 0 | 0 | 146.0 (5) | 421.3 | 147.5 (15) | 425.1 |
| 1 | -2 | 1000.0 (0) | 3284.3 | 759.1 (4) | 2446.8 |
| 1 | 0 | 523.2 (5) | 1816.0 | 332.2 (15) | 1154.9 |
| 2 | -2 | 1000.0 (0) | 3312.5 | 110.1 (15) | 270.3 |
| 2 | 0 | 1000.0 (0) | 3286.8 | 112.2 (15) | 277.3 |

Removing actor risk did not rescue seed0 and harmed seed1. Across three training
seeds, beta=0 reduced deterministic mean length/return by 159.9/500.7 and sampled
mean length/return by 142.2/430.0. Keep beta=-2 as the continuation control.

Seed2 sampled length fell from 1000 at 300k (v124 matched CPU protocol) to about
110 in both 350k arms, with angle termination in every sampled episode. Its
deterministic policy still ran 1000 steps. Control final alpha was 0.0697 versus
0.0706 at 300k; on sampled rollout states, mean pre-tanh policy std/action-noise
L2 rose from 0.1577/0.1736 to 0.1881/0.2117. Training still had only one physical
termination in the last 5000 steps, so rollout context, noise sensitivity and
inference-device effects need separating before an entropy intervention.

Next: v126 frozen diagnosis, submitted; task state is recorded in
`experiments/general_ood_v126_noise_sensitivity_20261011/dispatch_manifest.json`.
Compare all three beta=-2 sources
at 300k and 350k under common resets and explicitly shared Gaussian noise,
with noise scales 0/0.25/0.5/1. Keep the existing reset/noise seeds for continuity;
also use a separate, predeclared fresh noise set to test whether the seed2 drop
persists. For seed2, compare CPU/GPU inference with the same noise schedule
and matched initial context, and record action/std/context plus physical tails.
No optimizer updates, reward/physics changes or readout refit. If device effects
dominate, address evaluation comparability first; otherwise use the noise-scale
response to choose a single matched training intervention. Do not yet expand
training seeds or reduce entropy merely because stochastic evaluation collapsed.

## Limitations

Three training seeds and a matched 50k continuation support this local decision,
not universal risk-term superiority or OOD claims. Noise repeats are not training
seeds. Rollout std/noise means use different visited states and episode lengths,
so their increase is observational, not proof of excessive entropy. The 300k/350k
comparison describes training evolution, not a matched-budget ablation. Periodic
GPU scores never replace final CPU scores; paper tables and past negatives remain
unchanged. Summary: `results/general_ood_v125_actor_beta/summary.json`.
