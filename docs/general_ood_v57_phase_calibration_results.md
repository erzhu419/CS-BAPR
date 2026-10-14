# v57 phase calibration

## Preflight

Task t136478 completed with exit0. The 61,217-byte scalar result matches all12 cells, frozen stage specs, paired prefixes, score signals and contrast calculations: 16,000 interactions/updates. Both phase references cover ages0--999.

Candidate normal full means are0.018199/0.032707; last64 means are0.000597/0. All remain below0.10. Every changed-pair first50/last64 contrast is positive (8/8 each), and larger than the same-data pooled control (8/8 each). That control passes7/8 early and8/8 tail signs.

Gravity98820: candidate early/tail0.012060/0.006452 versus matched pooled0/0.000503. Gravity98821: candidate0.045377/0.008708 versus matched pooled0.000883/0.000294. Frozen v55 passes all signs on these seeds too; its early contrasts are slightly larger, while its tail contrasts are smaller. The intervention is not uniformly superior to every reference.

## Formal
Task t136481 completed with exit0. All22 cells, paired prefixes, frozen scores, phase coverage and contrasts match the protocol: 28,000 interactions/updates; scalar JSON93,034 bytes. Candidate early/tail contrasts pass16/16 each and exceed same-data pooled controls16/16 each. Frozen v55 tails pass15/16; matched pooled tails pass14/16.

Candidate normal full means are0.003496/0.018991/0.016138/0.005447; last64 means are0.000469/0.005676/0/0. Gravity tail contrasts for98920--98923 are0.030084/0.078939/0.067844/0.026160. The predeclared detector gate passes on independent continuation/test seeds.

## Limitations
Normal coefficients increase relative to the pooled control, within the original bound. Frozen v55 has larger early contrasts on these trajectories; the improvement is not uniform across references. All cells start from one source-training seed at fixed beta=-2. Adaptive RL and50x OOD return benefits remain untested; v55 formal remains HOLD.

## Next
v58 matched closed-loop pilot: fixed beta=-2, fixed beta=-2.3, pooled beta and phase beta; two new continuation seeds, 6000 steps per arm. Frozen v57 formal references, identical initialization and training budget, actor-only beta=-2-risk. Evaluate ID, original four profiles and50x Markov OOD before any performance expansion.
