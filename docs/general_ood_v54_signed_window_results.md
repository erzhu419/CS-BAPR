# v54: stronger tails, one early gravity tie

t136167 completed14 cells,16800 interactions/14400 updates, exit0, archived.
The40403-byte JSON matches split, budget, prefix, contrast and summary contracts.
Normal mean coefficients are .015009/.010357; last64 .014954/.000718.

| Profile | First50 paired differences, seed98420/1 | Last64, seed98420/1 |
| --- | --- | --- |
| heavy | .012435 / .063995 | .069900 / .169559 |
| slippery | .164482 / .213240 | .329278 / .467390 |
| high_grav | .000000 / .000205 | .004150 / .080827 |
| compound | .074875 / .051791 | .200969 / .196667 |

All8 tails are positive. Only gravity seed98420 first50 ties its control: both
coefficients .00035865, despite signal .107590 versus .052897. The predeclared
gate remains HOLD; no formal expansion or adaptive RL dispatch.

Next v55 freezes v54 scores/covariance and compares the pooled threshold with an
age threshold calibrated only on new normal online-learning trajectories. Replay
the old calibration to inspect frozen/learning contributions, then test unused
seeds. Keep the criterion, physics, loss, window and global score scale unchanged.

Limitations: one source seed, fixed beta=-2, conditional detector probes; no RL
return or50x OOD confirmation. The tied early response is not a negative effect.
