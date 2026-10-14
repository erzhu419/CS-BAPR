# v143 terminal-window results

All eight formal tasks t148331--t148338 completed exit0, plus three native CPU
smokes.160 formal episodes /37289 steps, zero training/optimizer updates.
Only4657715 bytes of formal window/scalar JSON pulled; checkpoints stay on
servers.33 recorder/source regression tests plus7 summary tests passed.

| Frozen source -> evaluation | Sampled mean length /15 | Falls /20 |
|---|---:|---:|
| ID seed4 -> ID | 1000 | 0 |
| ID seed4 -> friction-only | 170.40 | 20 |
| ID seed4 -> gravity-only | 90.93 | 20 |

The five failed matching-profile models reproduce the v142 survival plateau.
All120 matching episodes retain exact lengths, termination types, reasons and
protocol. ID4 evaluated on jtl110gpu has floating trajectory differences from
the archived node007 evaluation: maximum return difference.194, terminal height
difference.0126m; all20 remain healthy through1000 steps. Differences are stored
in summary.json, not silently treated as bit-identical trajectories.

Failing models do not saturate all actuators continuously. Sampled ankle-command
saturation is6.2--39.6% for failed matching models, versus33.9% for successful
ID4. Saturation alone does not distinguish failure from success. Matching failed
models finish with negative knee angles and ankle positions around the lower
joint stop, while stable ID4 sustains a different posture. These are observed
control/posture patterns, not a causal actuator-capacity certificate.

Next: a native fixed-profile standard-SAC positive control, same training seeds,
profiles, budgets and final evaluation. Stop extending the present controller;
first distinguish learnability under these physics from its learning failure.

## Limitations
Two original training seeds; reset/noise repeats are not independent seeds.
One stable ID policy fails across both altered physics. This does not prove
physical impossibility, algorithm superiority, or identify a unique failure cause.
