# v62 causal innovation results

t136541 completed on node007/GPU1, exit0. All16 cells,96000 interactions,
72000 updates and the inherited v60 reference match the protocol. All10
updating prefixes match; paired contrasts and gates reproduce from the JSON.

| Readout | Largest gated normal mean | Positive first50 | Positive tail64 |
| --- | ---: | ---: | ---: |
| v60 frozen | 0.076956 | 8/8 | 7/8 |
| Level recalibrated | 0.073567 | 7/8 | 7/8 |
| Level plus innovation | 0.079068 | 7/8 | 7/8 |

All remain HOLD. Both new readouts miss the seed134020 gravity first50;
all three miss the seed134021 gravity tail. Adding innovation does not
improve the pass counts relative to identical-data level recalibration.

In the early gravity failure, joint mean signal rises from normal0.200819
to0.344081, but both instantaneous and smoothed risk are0. Innovation dominates
only8% of the changed window. In the tail failure it never dominates;
level mean signal is0.134790 versus normal0.073573, with both risks0.
Innovation can decay after a persistent change, so it does not by itself
provide lasting regime evidence.

Next: v63 performs server-only replay attribution of threshold margins,
signed residual displacement and input support, including old v61 negatives.
No new training, threshold sweep or closed-loop expansion.

## Limitations

Two continuation seeds share source training seed1. These are fixed-beta
development readouts under mild physical changes, not adaptive RL gains or
extreme50x confirmation. The new readout has not passed its original gates.
