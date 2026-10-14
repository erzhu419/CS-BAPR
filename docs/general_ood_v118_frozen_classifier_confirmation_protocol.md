# v118 frozen-classifier confirmation

Before new outcomes, freeze the formal v115 final neural mean, v116 final matched
classifier, v113 final original classifier and all three v117 normal-null phases.
Nearest mean shares the same neural prediction. Retain all three v98 density
conditions and their original phases; primary incumbent is suffix_disagreement_mix.
No refitting, calibration, temperature/hazard search or candidate selection.

Fresh environment seeds: 135260/135261/135262; smoke: 135259. Assert separation
from every v117 development seed, mean-reference source and normal-null source.
Each seed has five modes x frozen/updating, 6000 steps/path, switch at 3800.
Use the unchanged v95 paired collector, source training seed1/update99745,
shared context/replay and fixed-minus2 continuation. All readouts observe the
same actual trajectories; none controls the actor. Five counterfactual simulator
steps per transition are retained by the collector but never used for scoring
or added to RL replay. Recursive history persists across episode resets.

Totals: 30 paths, 180000 actual transitions, 900000 shadow steps, 1080000 total
environment calls, 90000 fixed-minus2 optimizer updates. No candidate training.
Native smoke reproduces six full-history v117 readouts on the old high_grav
regression path, then checks 60 fresh path/readout combinations, paired prefixes
and causal prefix agreement with the full formal models and inherited phases.

Judge changed-minus-normal risk separation and normal behavior jointly, not just
exact changed-mode identification. Report all early/tail accuracies, minimum
departure contrasts, normal risk means, worst normal episode means and measured
threshold exceedance for the candidate and incumbent. Keep adverse seeds/modes.
Summarize paired updating candidate-minus-v98 early/tail accuracy and normal
risk/exceedance with n=3 environment-seed t intervals; do not treat steps as
independent repetitions. These diagnostics inform a later small actor-feedback
test against v98 and fixed-minus2, not automatic adoption.

## Limitations

One source policy and known mild Hopper profiles; new environment seeds do not
replicate training seeds or environments. Empirical score weights are not
physical-density Bayesian posteriors. Normal exceedance is measured, not a
guaranteed false-positive rate. Only scalar JSON returns locally; checkpoints,
arrays and traces stay native. No actor-return or manuscript-table update.
