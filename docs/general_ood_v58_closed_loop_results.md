# v58 closed-loop pilot: mixed returns, late normal drift

All eight tasks t136516--t136523 completed with exit0. The eight scalar JSONs
(252,575 bytes) match initialization, reference metadata, first transition,
within-seed mode schedule, final checkpoint selection and paired evaluation
seeds. Each arm used6000 interactions/updates:48000 total, plus88561 evaluation
interactions and zero evaluation updates. Actor beta accounting is correct.

| Seed | Arm | ID | Slippery | Compound | Markov50x |
| --- | --- | ---: | ---: | ---: | ---: |
|116000|fixed-minus2|1042.39|610.17|748.98|546.06|
|116000|fixed-minus2p3|1047.17|1044.43|1046.75|723.41|
|116000|pooled-beta|1038.77|961.87|1036.89|719.81|
|116000|phase-beta|1043.39|1047.13|1042.65|710.46|
|116001|fixed-minus2|1039.75|1002.34|994.78|797.03|
|116001|fixed-minus2p3|1045.08|1036.29|1047.07|572.95|
|116001|pooled-beta|1014.51|977.17|1031.09|806.15|
|116001|phase-beta|1033.87|1034.41|1034.13|810.19|

Phase beta improves Markov50x over beta=-2 by164.40/13.16, but differences
against beta=-2.3 are-12.95/+237.24 and against pooled beta are-9.35/+4.04.
Heavy and high-gravity means remain1034--1050 across all arms. The slippery and
compound improvements also occur with the static penalty. Adaptive superiority
is not established; the seed116001 static control has a substantial50x failure.
ID differences versus beta=-2 are+1.00/-5.88, versus beta=-2.3 are-3.78/-11.21.

The final1000 training transitions are all normal physics. Phase coefficients
are0.459573/0.467543, yielding beta=-2.459573/-2.467543. With no feedback at all,
the fixed-minus2 arm's phase observers also report0.405509/0.229590. Thus the
late signal cannot be attributed solely to adaptive beta feedback. Whether it
reflects ordinary continued learning or behavior learned during OOD is open.
The residual and departure windows are each16 steps; the final normal block
starts200 steps after the last mode change, excluding direct old-window carryover.
Pooled feedback is also inconsistent: late normal coefficients0.536474/0.033223.

## Limitations
Two continuation seeds share source-training seed1, and each condition has only
two evaluation episodes. This is a short development pilot, not independent
training confirmation. v57's short fixed-beta detector pass does not establish
a usable normal null over6000 steps. No paper claims or older tables changed.

## Next
v59 normal-only attribution: reuse116000/116001 as paired counterfactual seeds,
with the exact same source, replay and frozen references. Compare zero versus
6000 optimizer updates at beta=-2, with6000 normal interactions in each branch.
Read pooled and phase sensors without feeding either into actor loss; report
early/late bins and parameter movement. If late drift occurs without OOD,
repair learning-distribution calibration before another adaptive expansion;
otherwise localize persistence after OOD exposure. No beta/return tuning.
