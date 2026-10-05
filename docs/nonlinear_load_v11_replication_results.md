# Nonlinear load v11: seed7/8 confirmation

Eight tasks completed with exit code 0. All frozen configurations, 100,000 learner interactions, 99,001 updates, runtime versions, saved fitted gains and paired paths 50000--50099 were verified. The eight evaluation JSONs total 969,096 bytes and contain 15,600 evaluation episodes including controls. No checkpoints were downloaded. Manuscript tables are unchanged.

| Method | ID mean cost | 50x mean cost |
| --- | ---: | ---: |
| Jacobian-only | 0.259403 | 4.033380 |
| No-symbolic | 0.409082 | 6.528946 |
| BAPR | 20.894759 | 1394.702583 |
| RE-SAC | 13.725123 | 483.781088 |

The new cohort alone passes every frozen gate: both seeds beat all three learners at ID and 50x, and both beat their own fitted feedback. At 50x, cost reduction versus no-symbolic/BAPR/RE-SAC is 38.22%/99.71%/99.17%. Learner-minus-fitted return gaps are +0.019725/+0.080415 for seeds 7/8; the average fitted-feedback cost reduction is 1.23% (ID: 0.53%). The corresponding 50x path wins are 80/100 and 100/100; seed7's worst gap is -0.473910 on path 50065.

## Limitations

This confirms new training seeds on the existing environment and evaluation paths, not general RL transfer. Most of the performance comes from staying close to the fitted feedback; the additional RL gain is small. At 50x the candidate still costs 47.94% more than the privileged known inverse-actuator control, which is not an optimality certificate. Only two confirmation training seeds were tested; development seeds 5/6 are not pooled into the confirmation decision. The stronger no-symbolic results on seeds 7/8 also show why the earlier much larger margin should not be treated as universal.

## Next Step

Keep the current candidate, fitting procedure, losses, training budget and environment frozen. Evaluate the saved seed7/8 policies and all three baselines on 100 unused paired paths at all six severities; include each candidate's own saved fitted feedback and both known controls. Apply the same ID/50x two-seed gates independently to the new paths, retain negative tail gaps, and do not retrain or refit. Keep evaluation on workers and return only small JSONs. No new tasks have been submitted in this completion turn.
