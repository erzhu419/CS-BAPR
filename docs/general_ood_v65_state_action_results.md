# v65 state/action screen

t136628 finished on node007 CPU, exit0, in110 seconds. Six frozen-runtime tests
passed. Added RL interactions/updates0;17 cells replayed102000 saved transitions.
Only446932 bytes of scalar JSON returned. Inherited controls, paired contrasts,
budgets and gate summaries match the saved results.

The main conditional-center/OAS-covariance candidate remains HOLD. v62 early
contrasts pass8/8, but tail contrasts pass6/8 and frozen-normal seed134021
first800 mean risk0.236122 exceeds0.10. v61 replay passes normal and early4/4,
but tail3/4. Failed tails: seed134020 gravity/compound (zero contrasts) and
seed132020 compound (-0.018244). Calibration mean OAS shrinkage is0.070239.

Center-only with unchanged global precision passes both groups' normal bounds
and every tail (v62 8/8, v61 4/4). Early counts are7/8 and2/4; failures are
seed134020 gravity (zero), seed132020 gravity (-0.091843) and compound (-0.056681).
It is an ablation, not the selected main candidate.

Next: estimate a global/local covariance mixing weight by blocked validation
on normal-fit data, with normal-calibration data reserved for score/threshold
calibration. Retain conditioning, gates and all controls. Do not expand RL
seeds or change environment strength on the basis of this failed screen.

## Limitations

Reused development trajectories and one source training seed; no independent
confirmation or adaptive RL gains. Neighbor-radius diagnostics do not establish
conditional support. Full references and raw data remain server-side.
