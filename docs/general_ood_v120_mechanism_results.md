# v120 diagnosis and next training round

All three CPU tasks t141749/50/51 completed with exit0. Reused60 native traces,
45000 source evaluation steps, zero optimizer updates; scalar sync279420 bytes.

The unchanged source's15 seed/mode mean velocities are0.04079--0.04746 m/s;
all source episodes survive1000 steps, returns1039.65--1046.32. This is a
survival-dominated source, not a useful locomotion baseline. Matched changed
training velocity=0.03976 m/s (v98=0.04371);86.1% of its post-switch steps have
absolute velocity below0.1. Healthy reward1 explains almost all return.

Feedback is not disconnected: matched changed risk=0.97010, normal=0.00183.
On common compound-tail inputs, a unit beta increment has raw gradient norm
0.24--0.81 times the beta=-2 actor-loss gradient for matched endpoints. Therefore
zero uncertainty gradient is not the explanation. Endpoint probes do not
identify historical updates or prove what causes poor locomotion.

Next training round: first train a locomotion source in normal Hopper with the
original reward/physics, before adding OOD switches. Start with the existing
fixed-minus2 context architecture and a matched training budget; inspect velocity
and survival rather than treating return around1000 as success. If the same
standing solution persists, isolate actor conservatism and reward/entropy scale
before changing environment difficulty. Do not retune the classifier or expand
feedback seeds on the current source. After useful locomotion is established,
rebuild source-matched references and repeat the frozen four-arm feedback test.

This training round is planned, not submitted. v119 remains a valid negative
result conditional on its weak source. One trained source, known mild modes,
reused development seeds and compound-only endpoint gradients limit the claims.
