# v129 results and next step

All three tasks t142719--t142721 exited zero and automatically synced 751098
bytes of compact JSON. Batch wall time: 4.44 minutes. The merger validated nine
frozen models / three training seeds / 945 episodes / 521148 evaluation steps,
including all 45 deterministic baseline matches. Zero training or optimizer
updates. Summary: `results/general_ood_v129_heldout_robustness/summary.json`.

Sampled results use full action-noise scale one, 20 new noise schedules and five
common resets: 100 episodes per model and training seed.

| Seed | Model | Mean length | Mean return | Falls / 100 |
|---|---|---:|---:|---:|
| 0 | Reference 300k | 146.15 | 412.07 | 100 |
| 0 | Control 350k | 149.58 | 430.12 | 100 |
| 0 | Mean retention 350k | 148.15 | 421.13 | 100 |
| 1 | Reference 300k | 788.64 | 2664.93 | 61 |
| 1 | Control 350k | 605.90 | 1939.45 | 45 |
| 1 | Mean retention 350k | 969.10 | 3208.07 | 9 |
| 2 | Reference 300k | 1000.00 | 3288.22 | 0 |
| 2 | Control 350k | 153.96 | 419.27 | 95 |
| 2 | Mean retention 350k | 955.71 | 3137.49 | 5 |

The matched 350k comparison confirms that v128's benefit extends beyond its
three-noise groups. Seed1 improves mean return on 19/20 noise schedules:
40 paired episodes change from falling to finishing, five still fall under both
arms, and four newly fall under retention. Its nine retention falls are height
failures at steps 407--984, using noise13002/13003/13010. Noise13003 worsens mean
return by 675.79; control finishes all five episodes while retention falls three
times. Noise13002 has more falls (3 -> 4) despite improved mean length/return.

Seed2 improves mean return on 19/20 schedules and repairs 90 paired episodes.
All five residual falls use noise13015, at steps 114--115 with an angle failure;
the control also falls on those five episodes. Noise13018 finishes under both
arms but retention return is 12.08 lower. The 300k reference finishes all 100
sampled episodes, so retention only partially prevents continuation degradation.
Seed0 still falls in every deterministic and sampled episode; all 100 paired
sampled returns are slightly lower under retention (mean difference -8.99).

## Next Step

Plan v130 independent replication, not a coefficient sweep: add fresh training
seeds 3/4 using the unchanged v122/v123 L1-off source protocol to 300k, then fork
each complete native source state into the v128 control and mean-retention arms
for 50k each. Keep coefficient one, beta=-2, weight_reg=0 and every training
seed's own predeclared final 300k reference. New training cost: 800000
interactions across two source paths and four continuation paths. No filtering
of weak sources or replacement with better periodic checkpoints.

Evaluate all five training seeds under one common final protocol: new resets
151000--151004, new noise schedules 14000--14019, scale one, CPU/native Hopper-v4,
1000-step limit, all three final models. Report matched 350k arms by training
seed; retain the 300k model as reference context. Planned evaluation: 1575
episodes. Keep snapshots/replay native, sync small scalar JSON only, no Windows
nodes or machine/GPU preferences. Submitted after both native smokes passed;
see `experiments/general_ood_v130_independent_replication_20261011/dispatch_manifest.json`.

## Limitations

Only three independent training seeds; noise/reset repeats are evaluation
coverage, not additional training seeds. Resets were reused in v129. Mean
retention preserves useful behavior but does not repair seed0, eliminate every
regression, or identify the training pathway through std/context. This is normal
Hopper source-policy evidence, not OOD superiority or submission readiness.
Paper tables and earlier negative results remain unchanged.
