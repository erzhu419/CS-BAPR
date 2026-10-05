# Nonlinear load v12: new-path confirmation

Eight frozen-policy tasks completed with exit code 0 in 1.8--5.0 minutes each. Configuration, saved fitted gains, runtime versions, old-path reproduction and all paired returns were verified. Formal evaluation covers 15,600 episodes on paths 60000--60099, without training or refitting. Evaluation JSONs total 849,310 bytes; only 751,704 bytes were pulled from remote workers. No checkpoints were downloaded and manuscript tables are unchanged.

| Method | ID mean cost | 50x mean cost |
| --- | ---: | ---: |
| Jacobian-only | 0.234204 | 3.607319 |
| No-symbolic | 0.373706 | 5.600672 |
| BAPR | 17.763856 | 230.667860 |
| RE-SAC | 13.743351 | 418.463670 |

Both seeds pass all frozen ID/50x gates on the new paths alone. At 50x, cost reductions versus no-symbolic/BAPR/RE-SAC are 35.59%/98.44%/99.14%. Own-fitted return gains are +0.034670/+0.065095 for seeds 7/8, averaging a 1.36% cost reduction (ID: 0.52%). Fitted-feedback path wins are 83/100 and 100/100; seed7's worst gap is -0.206371 on path 60097. This agrees with v11's 1.23% mean improvement on old paths, without pooling the two cohorts for a gate.

## Limitations

These are the same two trained policies evaluated on new disturbance paths, not additional independent training seeds or a new environment. RL's extra improvement over its fitted reference remains small. The candidate still costs 40.78% more than the privileged known inverse-actuator control at 50x; that control is not an optimality certificate. BAPR's 50x mean changed substantially between path sets, so its precise improvement percentage is path-sensitive even though both matched-seed comparisons pass.

## Next Step

Keep the current policies, fitted references and disturbance process frozen. Test physical-state reset radii 0.5 and 1.0, alongside the original 0.2, at ID and 50x on the same paired new paths. Compare all four learners and direct controls separately at each radius; retain any failure rather than changing training, fitting or losses. This tests whether the current result depends on a small regulation neighborhood before expanding to a different environment. No new tasks have been submitted yet.
