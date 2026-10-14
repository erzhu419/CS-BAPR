# v61 early-response localization

t136538 completed on node007/GPU1 with exit0. The small JSON reports all five
seed132020 reruns exactly matching v60 first50/tail risk and signal (maximum
error0); all paired prefixes match. The score decomposition error is below
2e-12. Budget:30000 interactions and30000 updates, fixed beta=-2.

First50 changed-minus-normal contrasts:

| Mode | Level score | Cross term | Displacement | Instant risk | Smoothed risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| heavy | 0.416002 | -7.626957 | 8.042959 | 0.152287 | 0.109477 |
| slippery | 0.466627 | -5.374882 | 5.841509 | 0.034682 | -0.007285 |
| high_grav | -6.192069 | -5.650949 | -0.541120 | -0.036707 | -0.036603 |
| compound | -4.107873 | -7.500018 | 3.392144 | -0.050576 | -0.051361 |

The level score can decrease when a changed residual moves towards the global
normal center. Compound has positive additional displacement but a larger
negative cross term; gravity has both negative contrasts. This is not evidence
that a squared local displacement alone will separate every change.

Slippery has a positive instantaneous first50 contrast but a negative smoothed
one. Its post0--20 smoothed contrasts are positive; post20--30 and30--50 are
negative. The same reversal appears in compound. Thresholds are identical
between paired cells in all reported windows, ruling out different threshold
selection as the explanation for these particular contrasts. The normal
control itself has a short first50 pulse of0.431029.

Next: v62 tests a causal difference between adjacent16-transition signed
residual windows alongside the persistent level score. Freeze the v60 dynamics;
calibrate the joint signal on separate new normal trajectories and test new
paired continuations, retaining the original normal/first50/tail gates.

## Limitations

This is exact localization on a reused development seed, not independent
confirmation. The v61 anchor uses a known switch boundary and cannot be used
as an online detector. No readout changed actions; no new RL return claim.
