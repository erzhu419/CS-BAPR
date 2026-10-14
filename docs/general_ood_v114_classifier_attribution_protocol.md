# v114: fixed-classifier attribution

Freeze the v113 **formal final** scaler and quadratic logistic classifier, the
34,000-row expanded mean reference, global forward models and target scales.
Recompute both original scores on all 76 existing windows (4,248 transitions)
and match all ten scalar statistics per window to the archived v113 result.
No fitting, collection, temporal replay or actor change.

At each transition decompose the true-mode minus best-other logit into the
standardized-feature intercept, 55 linear residual terms and 55 squared residual
terms. Also retain contributions grouped by reference hypothesis, competitor
frequencies, and true-minus-normal decomposition for changed modes. Check that
the terms reconstruct the full margin. The intercept is not a mode prior.

For the 74 oracle-capable windows, replace all five predicted outcomes with the
simulator outcomes at the same observed state/action. Keep the classifier and
observed next state fixed. Check the true-mode oracle against the actual next
state and normalized target. Report classifier and nearest-mean scores for both
center arms, with paired oracle-minus-fitted summaries. Two historical v62
windows retain only original-score checks. Cohorts include all current failures,
normal controls, matched frozen controls, six earlier failures and all frozen
changed windows, including the largest v113 regressions.

Native smoke uses seven windows (364 transitions): seed135252 updating normal,
slippery, high_grav and compound; seed135191 frozen slippery first50;
seed135192 updating high_grav first50; historical seed134020 updating slippery
last64. It loads the full formal classifier, not a partially trained smoke model.

## Interpretation

Oracle rescue indicates sensitivity to mean inputs, not a deployable estimator.
Persistent error with correct centers points to the frozen scoring rule on those
inputs; it does not prove a particular remedy or a Bayes-optimal ceiling.
Feature magnitudes are descriptive, not training-support certificates. Best
competitors are chosen separately in each arm; fixed true-minus-normal gaps
provide an additional comparison when the best competitor changes. Previously
inspected evaluation windows remain development evidence.

## Execution

CPU-only scheduler jobs use node007 because all native input arrays and models
already reside there. No Windows placement or soft node/GPU preference. Stage
code only; return scalar JSON only. Formal dispatch follows local tests and
successful native smoke. No seed expansion, adopted readout or manuscript change.
