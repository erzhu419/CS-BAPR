# Nonlinear load v15: coverage intervention rejected

All ten formal tasks t124924--t124933 exited 0. Every run completed 500 episodes, 100,000 learner interactions and 99,001 updates. All 20,400 controller evaluation episodes passed protocol, paired-path and cost-accounting validation; matched candidate fits and known-control returns agree across runs. The existing 12,660,419 bytes of result JSONs were used locally; no additional remote data or checkpoints were pulled for this analysis.

| Radius 1.0 controller, two-seed mean cost | ID load | 50x load |
| --- | ---: | ---: |
| Candidate, narrow training resets | 5.306107 | 8.686836 |
| Candidate, balanced training resets | 5.433458 | 8.791397 |
| No-symbolic centered MLP, balanced | 9.026779 | 12.800009 |
| BAPR, balanced | 31.837187 | 782.002145 |
| RE-SAC, balanced | 40.502755 | 413.755463 |
| Own fitted feedback | 5.181082 | 8.613156 |

Coverage repair fails on both seeds at both severities: total cost rises 2.40%/1.20%, and first-50-step state cost rises 0.193254/0.181876 (ID/50x). At radius 1.0/50x, initial physical-input action-Jacobian squared mismatch increases from 0.105148 to 0.208875, while command effort falls. Expanding reset coverage alone did not correct initial regulation.

The candidate beats BAPR and RE-SAC at every radius/severity on both seeds. It also beats the no-symbolic MLP at radii 0.5/1.0, but loses radius 0.2/50x on seed9 by 0.466183 mean cost. Own-fitted added value passes only at radius 0.2; it fails at 0.5 and 1.0. Full confirmation remains HOLD, without pooling seeds or changing gates. The historical v13 HOLD is unchanged.

## Next Step

Stop the reset-coverage intervention. Use the four existing candidate checkpoints for read-only, state-matched SAC/Jacobian gradient decomposition and replay-state support diagnostics, contrasting initial states with replay/late-trajectory states. Determine whether optimization conflicts or training-sample weighting explain the larger initial Jacobian mismatch before changing losses or environments. No optimizer updates, new seeds or new tasks have been submitted.

## Limitations

These are two fresh training seeds in one observed-load environment, using previously evaluated disturbance paths. Balanced radii are covered training resets, not unseen-state OOD. The gradients and replay-distribution mechanisms above remain hypotheses, and the direct feedback controls are not certified optima.
