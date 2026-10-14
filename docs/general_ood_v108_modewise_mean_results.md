# v108: hard mode-wise selection leaves a tradeoff

Task `t141231` completed in 6m01s: 95 paths / 570,000 cached transitions,
570 old/new path-condition comparisons and six calibration comparisons passed.
No environment interactions, actor updates or large-artifact pulls were added.

Primary condition: `suffix_disagreement_mix`; equal early/tail window weights.

| Metric | Old | New | Old normal, new OOD | New normal, old OOD |
| --- | ---: | ---: | ---: | ---: |
| v95 updating recognition | 68.12% | 74.27% | 72.19% | 68.85% |
| All updating recognition | 81.39% | 84.69% | 83.87% | 81.50% |
| Six old failures: recognition | 14.44% | 30.55% | 22.89% | 16.51% |
| Maximum updating normal-window mean risk | 0.00438 | 0.03205 | 0.00942 | 0.06224 |
| Historical slippery tail: recognition | 32.81% | 65.63% | 65.63% | 32.81% |

Old normal plus new OOD repairs most seed135192 normal confusion (accuracy
95.22% to 98.20%, versus old 98.58%) but loses early OOD recognition. It also
raises another updating normal window: seed135181 first800 risk 0.00190 to
0.00942. Its v95 maximum normal risk is 0.00651, not the all-updating maximum.
The all-normal maximum is 0.01086 versus old 0.01002. All v95 frozen changed
accuracies remain 98.75%. The other covariance conditions show the same tradeoff.

**Decision:** retain v98. Next test fixed equal-weight old/new density mixtures
for normal only and for every mode, calibrated separately. This preserves model
ambiguity rather than choosing one reference mean per mode; v109 records execution.

**Limitations:** development-only paths, one source policy, unequal reference
budgets and reused normal calibration. Risk means are not false-positive rates;
recognition gains do not establish actor or return gains.

Artifact: `results/general_ood_v108_modewise_mean/formal/modewise_mean.json`.
