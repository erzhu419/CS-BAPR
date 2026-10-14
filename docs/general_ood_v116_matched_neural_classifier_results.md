# v116 results and next step

Formal t141578 completed, exit 0, scheduler wall time 370.05 s. All 18 source
folds used 100 epochs with the intended whole-source exclusions. Both classifiers
converged (outer 698 / final 659 iterations). All 76 v115 score controls matched.
Only 153,683 bytes of formal JSON returned locally; no arrays or models pulled.

Two-source development holdout: matched classifier 94.65%, original classifier
on the same neural means 93.65%, nearest mean 94.20%. Both sources improve, but
22,000 correlated labels are not 22,000 independent experimental repetitions.

Equal-window instantaneous accuracy, same frozen v115 means:

| Population | Original classifier | Matched classifier |
|---|---:|---:|
| Current four failures | 64.00% | 70.50% |
| Earlier six failures | 75.29% | 70.59% |
| Current four matched frozen | 87.50% | 90.50% |
| All 28 frozen changed | 91.93% | 94.97% |
| All 19 normal | 96.54% | 98.68% |

Normal recovery is partial: the pre-neural v113 classifier reached 99.54% on
these normal windows. Seed135192/high_grav first50 regresses 90% -> 56%.
Seed135252/compound improves 22% -> 52%, but its mean true-minus-best-other score
remains -0.117; seed135190/compound first50 is 28% with score -0.812. Both mainly
confuse compound with slippery. Across all windows, 30 improve, 11 regress and
35 are unchanged. Do not adopt v116 as an overall replacement.

## Next step

v117 should freeze the matched classifier, original classifier and nearest-mean
readouts for causal full-history recognition-only replay and reference-only
normal-null calibration under the same temporal rules. Evaluate whether pointwise
errors create persistent belief errors and normal false alarms before another
score redesign. Retain every existing control, especially both negative-margin
compound cases and the high_grav regression. Positive mean margins on 27/29
updating changed windows motivate this diagnostic, but do not guarantee recovery.
No new data, seed expansion, actor change or automatic adoption. Not dispatched.

## Limitations

Source holdouts and inspected query windows remain development evidence. Mean
models are fixed; global forward models and input/target scales are inherited.
Classifier probabilities are not yet normal-null calibrated temporal evidence.
No full-history belief, actor or return improvement is established. v98 remains
retained.
