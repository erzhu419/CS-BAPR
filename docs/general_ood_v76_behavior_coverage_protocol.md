# v76 Behavior Coverage

Test the v75 coverage hypothesis with matched frozen/updating reference samplers.
Both run on the same device, source/shared start, seeds 135170/135171, five known
mode templates and fixed beta -2. Reference trajectories visit normal/slippery
profiles; their first 3,800 transitions remain normal. Each condition uses every
second post-switch transition from all four paths: 4,400 labels per mode, with
equal normal/slippery profile weight and identical time sampling.

Controls are the original v72 bank, the saved v74 slippery bank, frozen mixed
coverage and updating mixed coverage. The two mixed banks append labels to the
same 18,600-row base. Forward coefficients, input/target scales, 64-neighbor
geometry rule, recursive switch hazard 0.02 and normal calibration population
stay fixed. Each mixed bank recalibrates normal thresholds on the same two saved
normal-null paths; its mean/covariance support changes together.

Freeze all references before fresh seeds 135180/135181, each tested on all five
modes for 6,000 steps with the switch at 3,800. Only real transitions enter RL
replay; the observer never controls the actor. Report early/tail mode accuracy,
normal risk, heavy/gravity early regressions, and neighbor support. Previously
inspected paths remain development replays, not fresh confirmations.

New budget: eight reference and ten test paths, 108,000 actual transitions,
540,000 counterfactual steps and 84,000 optimizer updates (24,000 reference;
60,000 test). Updating reference collection has additional training cost even
though label budgets match. Verify native GPU updates, exact true-mode replay,
paired prefixes and resume before dispatch. Only scalar JSON/logs return locally.
