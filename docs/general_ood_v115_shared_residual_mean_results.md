# v115 results and next step

Formal t141570 completed, exit 0, scheduler wall time 60.85 s. Both mean fits
used 100 epochs (5,800 / 6,700 updates). All 76 original-score controls matched.
Four local tests passed. Only 253,072 bytes of formal JSON returned locally.

Complete-source development holdout (4,400 paired inputs): MSE 0.01773 -> 0.01149
(-35.2%); nearest-mode accuracy 86.62% -> 94.20%. Both sources and all five modes
improved on both metrics. Training inputs were matched at 29,600; final fits
used 34,000. No query-window tuning or classifier fitting occurred.

Equal-window instantaneous accuracy of the frozen v113 classifier:

| Population | Local-linear means | Shared residual means |
|---|---:|---:|
| Current four failures | 15.50% | 64.00% |
| Earlier six failures | 54.23% | 75.29% |
| Current four matched frozen | 75.50% | 87.50% |
| All 28 frozen changed | 87.17% | 91.93% |
| All 19 normal | 99.54% | 96.54% |

Current failures' mean true-mode RMS fell 0.2961 -> 0.1466. However,
seed135252/compound remains at 22% classifier accuracy (nearest mean 36%).
Normal seed135192/updating first50 falls 98% -> 72% with the classifier despite
nearest-mean recognition rising 98% -> 100%. Seed135191 normal similarly falls
100% -> 78%, while nearest-mean recognition rises 92% -> 100%.

## Next step

v116: freeze v115's trained means and build matching whole-source-held-out neural
prediction errors. Refit the same quadratic classifier on those residual features,
with eight-source nested training for the two-source development holdout and
ten-source cross-fitting for the final classifier. Keep the original classifier
and nearest-mean controls and all 76 query windows. No new data, hyperparameter
search, seed expansion or actor adoption. Normal controls must be assessed before
full-history belief replay. v116 is planned, not dispatched.

## Limitations

The source holdouts and query windows are development evidence. Frozen classifier
input interventions are not recalibrated likelihoods. The global forward model
and scales remain inherited. No full-history belief, actor or return gain is
established. v98 remains retained; v115 is not adopted.
