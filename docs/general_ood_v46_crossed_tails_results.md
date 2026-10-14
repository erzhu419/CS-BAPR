# v46: Whole-trace crossed replay

Task `t135920` completed on node007 GPU0 in 462.92 seconds. Same 24 normal
references and 200 frozen-policy test rollouts, 380,928 interactions, zero RL
updates/replay writes. Both source controls reproduce v44/v45; 25 tests pass.
Only 9,041,957 bytes of scalar posterior JSON were synced, no checkpoints.

Normal full1536 cases containing reset probability >.5, out of 20 per policy:

| Frozen policy | Episode/original | Episode/robust | Global/original | Global/robust |
|---|---:|---:|---:|---:|
| fixed-minus2 | 20 | 15 | 17 | 0 |
| relative-adaptive | 6 | 0 | 5 | 0 |

Global/robust retains positive early16 paired lambda response in all 160 changed
cases, with mean differences .360-.362 on fixed and .317-.454 on relative.
No first16 physical endings, no major resets after first16 in changed rollouts;
late64 profile-mean age recovers to 18.56-18.98. Normal whole-trajectory mean
lambda is .1689/.1786 for the two policies; relative normal late64 is still
.2357 versus .2092 for episode/original, so quiet hard-reset counts do not
imply uniformly smaller soft penalties.

## Next Step

Carry global/robust forward, keep episode/original as control. Freeze settings
and test normal->OOD->normal plus OOD->another OOD. For each tested switch,
recollect changed and stay-current-mode branches from the same seed and mode
history, requiring identical pre-switch trajectories and inputs. Retain soft
response, sham events and recovery; no extra RL seeds or actor training yet.

## Limitations

This is reused single-switch development replay, not independent confirmation,
RL return improvement or validation on the 50x benchmark. Heavy/gravity response
on the relative source can be soft without a majority reset; later windows
include physical failures. Full four-arm statistics: `results/general_ood_v46_crossed_tails/summary.json`.
