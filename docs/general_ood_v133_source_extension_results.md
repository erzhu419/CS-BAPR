# v133 results and next step

All four tasks t143349/t143350/t143352/t143353 completed with exit0 and synced
328601 bytes of source JSON. Each continued its own100k full state to300k:
800000 new interactions/updates, plus400000 reused prefix interactions.
Native training/evaluation took59.0--67.2 minutes per task. Original prefix
blocks/evaluations, configs and runtime versions match; checkpoints stay on
servers. Summary: `results/general_ood_v133_source_extension/summary.json`.

| Seed | Loss | Return at300k | Velocity | Length at100k ->300k | Falls /5 | Last5k mean TD MSE |
|---|---|---:|---:|---:|---:|---:|
| 3 | MSE | 3550.33 | 2.5510 | 613.6 ->1000 | 0 | 4.1036 |
| 3 | 2xHuber10 | 442.67 | 2.0079 | 139.0 ->147.2 | 5 | 1.0398 |
| 4 | MSE | 3322.28 | 2.3234 | 165.0 ->1000 | 0 | 5.4776 |
| 4 | 2xHuber10 | 3020.78 | 2.3178 | 268.2 ->914.0 | 2 | 37.5510 |

MSE wins both final seed pairs on return and survival; all10 deterministic
MSE episodes reach the1000-step limit with substantial forward motion. Keep
MSE as the working source route. Huber10 is not a successful stability repair:
seed3's low TD error coexists with short survival, while seed4 retains large
squared residuals despite its smaller optimized Huber objective3.7313.

## Next Step

Plan v134, frozen evaluation only, all four final300k actors retained. Per
actor: repeat the five deterministic anchor resets150000--150004, add20 reset
seeds151000--151019 in deterministic mode and with three full-scale action
noise seeds134000--134002. Total340 episodes, zero optimizer updates and no
replay required. Restore the source context normalizer before every episode,
reset only the short EMA, then update from actual transitions. Use the native
CPU evaluator/common noise convention; report CPU anchors alongside the GPU
anchors without assuming identical long trajectories. Assess normal sampled
survival before OOD or baseline-training expansion. Planned, not dispatched.

## Limitations

Only two training-seed pairs and five final reset seeds per actor are present.
MSE seed4's periodic lengths change1000 at250k ->131.7 at275k ->1000 at the
predeclared300k endpoint; robust survival under more resets/action noise is not
yet established. New training time limits are42/0/77/68 for seed3 MSE/Huber,
seed4 MSE/Huber, so corrected masks now operate, but there is no matched
incorrect-mask arm to identify their causal effect. No OOD superiority or
general Huber rejection follows. Block metrics are update means, not peak
certificates. Historical actors and paper tables remain unchanged.
