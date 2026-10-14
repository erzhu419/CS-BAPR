# v149 warm recovery results

All four native GPU tasks completed (t152103--t152106, jtl110gpu2).
Two v147 no-pulse600k source seeds were forked into matched control and
randomized-recovery arms. Each arm restored the full source state and added
100k interactions/updates. All 1440 evaluation episodes and 1280 pre-shock
pairs passed protocol validation; only four small JSON files (12,067,953
bytes) were pulled. Training took 1953--1962 seconds per task.

## Final700k

Sampled-action results below are mean episode length / healthy1000-step
finishes out of15. These are evaluation reset/noise repeats, not15 training
seeds. Physical shocks use unchanged held-out2.6591479484724942x gravity.

|Source seed / arm|No shock|after100,10 steps|after300,10 steps|after100,25 steps|after300,25 steps|
|---|---:|---:|---:|---:|---:|
|5 control|1000 /15|462.7 /4|613.3 /3|211.2 /0|351.5 /0|
|5 randomized|1000 /15|513.1 /4|1000 /15|169.7 /0|351.9 /0|
|6 control|1000 /15|772.4 /10|1000 /15|389.3 /3|1000 /15|
|6 randomized|1000 /15|295.7 /0|1000 /15|231.3 /0|1000 /15|

Both50-step and permanent shocks yielded zero healthy full finishes in all
four policies, at either onset. Warm starts preserved final ID performance
with sampled mean returns3448--3524 (source600k:3384/3424),
but did not produce consistent extreme-shock gains. Seed5 randomized early
10-step survival fell from14/15 at650k to4/15 at700k; seed6 had transient ID
degradation at650k in both arms. Both predefined checkpoints remain reported.

All four policies recovered100 steps after the early10-step shock in all15
sampled episodes, yet subsequently fell in11,11,5,15 episodes respectively.
Short recovery is therefore not equivalent to full-episode robustness.

## Noise Timing Lead

Seed6 randomized final700k survived the early10-step shock in5/5 deterministic
episodes, versus0/15 sampled episodes; its control survived5/5 versus10/15.
This motivates locating the action-noise effect in time. Global determinism
is not a universal solution: seed5 control late10-step shock had0/5
deterministic finishes versus3/15 sampled finishes.

Training randomized arms saw54/55 pulse episodes,540/550 peak transitions
and47/44 healthy pulse-episode finishes. Peak transitions constitute only
0.54/0.55% of new interactions and about0.077/0.079% of final replay; replay
dilution is a plausible explanation, not an established cause.

## Next Step and Limits

v150 freezes all four final700k policies and compares native noise, globally
quiet actions, and quiet prefix/pulse/recovery/pulse-plus-recovery windows.
Use new reset/noise seeds and same-run no-physical-shock controls for each
program. Keep gravity, native reward/termination and motor limits unchanged.
The true shock clock is an oracle diagnostic, not an implementable controller.
Only if a repeatable time-local effect survives the no-shock control should
we design a causal, observation-based noise adaptation experiment. Otherwise
return to mean-policy/recovery-state coverage, rather than adding context
or symbolic penalties without evidence.

These are two paired source seeds and standard-SAC calibration experiments,
not CS-BAPR mechanism evidence or a general-environment success claim.
