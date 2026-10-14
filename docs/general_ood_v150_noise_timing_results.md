# v150 frozen noise timing results

All four formal CPU tasks t152176--t152179 exited0 on node007. All1200
episodes,1,021,785 evaluation steps,640 physical-prefix pairs and960
noise-boundary pairs passed validation. No training, optimizer updates or
replay loading. Only8,668,415 bytes of formal JSON returned locally; final700k
ZIPs remained server-side. Native evaluation took204--254 seconds per model.

All ten no-physical-shock controls per model finished every episode. Fresh
reset/noise seeds differ from v149; evaluate v150 contrasts within v150,
rather than attributing cross-round differences to a training change.

Healthy1000-step finishes under unchanged2.6591479484724942x gravity,10-step
pulse after100 completed steps:

| Policy | Native | Globally quiet | Prefix quiet | Pulse quiet | Recovery quiet | Pulse + recovery quiet |
|---|---:|---:|---:|---:|---:|---:|
| seed5 control | 0/15 | 5/5 | 4/15 | 4/15 | 2/15 | 4/15 |
| seed5 randomized | 0/15 | 1/5 | 3/15 | 3/15 | 0/15 | 2/15 |
| seed6 control | 11/15 | 5/5 | 15/15 | 11/15 | 12/15 | 11/15 |
| seed6 randomized | 3/15 | 5/5 | 15/15 | 0/15 | 0/15 | 0/15 |

Prefix quiet restores sampling from the first peak-gravity action onwards.
For seed6 randomized, its15/15 finishes therefore do not require quieter
actions during the shock or recovery. This locates a pre-shock trajectory
dependence; it does not establish that responding to a detected shock with
less noise would help. The latter windows actually regress in this cell.
Seed5 improvements are incomplete, so no single time-local rule is universally
successful. Global determinism is also insufficient: seed5 control late
shock has0/5 finishes, versus native1/15.

After300-step shocks, seed6 finishes every episode under every program;
seed5 randomized native13/15 rises to15/15 for prefix/recovery/combined quiet,
whereas seed5 control remains only0--3 finishes per program. The source/gait
and onset interaction remains important.

## Next Direction and Limits

Before a learned noise controller, isolate pre-shock physical state, gait
phase and action-noise sensitivity using these frozen policies and matched
states. Any deployment rule must use observable history, not the true shock
clock. No new experiment was submitted during this handoff turn.

This is an oracle-clock standard-SAC diagnostic, not CS-BAPR, context-learning
or symbolic-learning evidence. Five deterministic resets and fifteen sampled
reset/noise repeats remain distinct; only two paired training-source seeds
are available. Summary: `results/general_ood_v150_noise_timing/summary.json`.
