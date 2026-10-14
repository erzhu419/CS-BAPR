# v135 results and next step

Tasks t144077--t144080 completed exit0 on node007 CPU; all three Linux nodes
were eligible without preferences. Two frozen final300k MSE policies,
480 episodes /85318 evaluation steps, zero training/optimizer updates.
Protocol and source/physics/context checks passed; 17 evaluator tests and
9 summary tests passed. Only669525 bytes of scalar JSON pulled locally,
including two smokes; no checkpoints/replay pulled. Older paper tables unchanged.

Mean sampled episode length (15 paired reset/noise runs per cell):

| Seed | Context | ID | Easy | Heavy | Mega | Slippery | Markov |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 | Online | 1000 | 272.00 | 103.73 | 74.33 | 109.33 | 148.87 |
| 3 | Zero | 12.33 | 12.33 | 11.47 | 11.00 | 11.93 | 12.33 |
| 4 | Online | 1000 | 140.00 | 103.33 | 72.13 | 111.40 | 127.33 |
| 4 | Zero | 448.13 | 122.47 | 101.87 | 68.93 | 110.20 | 104.27 |

Online survives all40 ID episodes, but all200 online OOD episodes physically
terminate. Seed3 zero-context fails before Markov's first switch: no OOD
exposure in that control. Its large difference therefore demonstrates context
dependence, not an isolated adaptation benefit. Seed4's sampled gains are
only1.47/3.20/1.20 steps in heavy/mega/slippery; Markov improves23.07 steps,
but all runs still fall. Deterministic easy-surface seed4 worsens3.60 steps.

Next v136: retain intensity50 and the same weights, but give both arms200
identical ID steps, then compare online context against context held at its
pre-switch value. Include an ID hold control plus four single abrupt-profile
changes at step201. Five new resets153000--153004, deterministic plus noise
136000--136002, two training seeds /two context modes:400 episodes, zero
training. Check identical pre-switch trajectories and report post-switch
survival/return separately. This removes startup failure from the adaptation
comparison without reducing OOD severity. Planned, not dispatched.

## Limitations

Two training seeds only. Zero/held context are inference interventions, not
independently trained baselines. v135 does not establish OOD adaptation,
physical infeasibility, symbolic-learning benefit or RE-SAC/BAPR superiority.
Summary: `results/general_ood_v135_frozen_ood_context/summary.json`.
