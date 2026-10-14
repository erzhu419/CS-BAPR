# v64 conditional vector screen results

t136608 completed on node007 CPU, exit0, in58 seconds. Added interactions and
RL updates0. Fit/calibration each used12000 saved normal transitions;17 replay
cells consumed102000 saved transitions. The245948-byte scalar JSON was the
only returned artifact. Six frozen-runtime tests passed; budget, paired
contrasts, inherited controls and gates reconstruct from the result.

| Group / geometry | Normal bound passes | Largest gated normal mean | First50 positive | Tail64 positive |
| --- | --- | ---: | ---: | ---: |
| v62 / center only | Yes | 0.071041 | 7/8 | 7/8 |
| v62 / center + covariance | No | 0.306158 | 8/8 | 7/8 |
| v61 replay / center only | Yes | 0.044146 | 1/4 | 4/4 |
| v61 replay / center + covariance | Yes | 0.071709 | 4/4 | 3/4 |

Main candidate remains HOLD. Conditional covariance improves early contrasts
relative to center-only, including seed134020 gravity (+0.041434). The old
seed132020 gravity contrast is only+0.003879. Normal frozen-policy first800
risks0.118246/0.306158 exceed the unchanged0.10 bound. Seed134021 gravity tail
and old seed132020 compound tail both have exactly0 paired risk contrast.

Center-only preserves the normal bound but does not resolve early failures.
This screen does not justify new RL runs or selecting an ablation as fallback.
Next: test conditioning on observed state/action rather than episode age alone,
and examine conditional covariance stability, using stored normal-fit/calibration
data before collecting new continuations. Keep all original gates.

## Limitations

All replay trajectories are reused development data from one source training
seed. Positive contrast counts are not independent confirmation or adaptive RL
gains. Full centers/precision matrices remain in the server reference checkpoint;
the trained dynamics, trajectories and beta were unchanged.
