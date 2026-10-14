# v134 results and next step

Tasks t144010--t144013 completed with exit0 on node007 CPU. All three Linux
nodes were eligible, without preferences; no GPU/Windows allocation. Four
final300k snapshots /340 episodes /252240 evaluation steps, zero training or
optimizer updates, no replay/CUDA initialization. Native evaluation took
11.8--71.7 seconds per model. Only272242 bytes of scalar results pulled locally.

| Seed | Loss | New deterministic length /20 | Sampled length /60 | Sampled return | Sampled time limits /60 |
|---|---|---:|---:|---:|---:|
| 3 | MSE | 1000.0 | 1000.0 | 3507.69 | 60 |
| 3 | Huber10 | 147.0 | 146.68 | 440.96 | 0 |
| 4 | MSE | 1000.0 | 1000.0 | 3286.03 | 60 |
| 4 | Huber10 | 942.35 | 772.52 | 2558.72 | 29 |

Both MSE sources survive all170 episodes, including120 native full-scale
sampled-action episodes. Sampled action-noise L2 averages0.214/0.187, so this
is not a deterministic evaluation mislabeled as stochastic. Normal-policy
readiness passes this fixed protocol. Huber3 remains short-lived; Huber4 loses
survival under sampling. CPU anchors largely reproduce GPU outcomes, with
Huber4 mean length913 versus914; the numerical difference is retained.

Next, plan v135 frozen OOD/context-ablation evaluation on both MSE sources:
existing structured-family intensity50, four static non-ID profiles plus
ID and Markov switching every50 steps with probability1. Keep original rewards
and observable-state interface, compare online versus zero context using the
same weights/reset/noise schedules. Five new resets152000--152004 and three
action-noise seeds135000--135002:480 episodes, zero updates. Log actual physical
multipliers, non-ID exposure, switches and failure causes. Planned, not dispatched.

## Limitations

Only two training seeds; reset/noise repetitions are not additional training
seeds. No OOD, context-mechanism or RE-SAC/BAPR superiority follows from v134.
Intensity50 is a profile-generation parameter, not50x every physical variable.
Historical actors/results and paper tables remain unchanged. Summary:
`results/general_ood_v134_frozen_source_robustness/summary.json`.
