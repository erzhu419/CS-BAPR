# v112: matched errors repair older failures, not current failures

Formal `t141511` completed, exit 0, in 176.44 seconds. All 29,600 held-out
errors were refitted across eight source seeds. The 76 windows / 4,248
transitions completed; 186 old-attribution and 84 cached-likelihood comparisons
passed. Only 1,342,134 bytes of scalar JSON returned.

Primary condition: `suffix_disagreement_mix`. These are equal-window mean
**instantaneous** accuracies, not complete-history belief or return results.

| Inspected population | v98 | Expanded means, old errors | Matched errors |
| --- | ---: | ---: | ---: |
| Earlier six failed windows | 20.51% | 29.48% | 73.06% |
| Current four failed updating windows | 8.00% | 10.00% | 9.00% |
| Current six normal controls | 94.33% | 91.67% | 89.00% |
| Current four matched frozen controls | 86.50% | 87.00% | 87.00% |

All six earlier failures improve versus expanded means with old errors. Earlier
updating changed windows improve 63.30% to 80.45%; earlier frozen changed windows
fall slightly, 97.03% to 96.78%. Current failures do not resolve under any of the
three covariance variants.

Seed135252 updating normal falls 68% -> 54% -> 36%; matched errors predict
heavy on 64% of its first 50 post-switch steps. For that seed's
slippery/high_grav/compound windows, matched-error accuracy is 0%/2%/0%.
Even diagnostic correct simulator centers give only 0%/8%/4%: energy advantages
+2.61/+1.15/+2.41 are outweighed by volume advantages -5.70/-2.29/-5.10.
Matching held-out errors is therefore not sufficient to repair these rankings.

**Decision:** HOLD; retain v98. No full replay, seed expansion or actor adoption.
The next proposed pilot is paired-counterfactual discriminative mode evidence:
balance all modes at identical reference state/actions, hold out complete source
seeds, and infer from only the observed real transition. First compare on the
same failure, normal and frozen windows. This alternative is not implemented
or dispatched; the result does not justify merely deleting Gaussian logdet.

**Limitations:** all inspected windows are development evidence. Oracle centers
are simulator-only diagnostics, not deployment or an identifiability bound.
The old disagreement moment remains fixed; only the error second moment changed.
Two historical v62 windows lack oracle observations but retain real-arm controls.
