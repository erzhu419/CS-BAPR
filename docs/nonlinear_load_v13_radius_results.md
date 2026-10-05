# Nonlinear load v13: reset-radius stress results

All eight tasks exited 0. Frozen configurations, saved gains, pinned CUDA runtimes, all 100 paired radius-0.2 actor/control returns, and radius-specific diagnostics were verified. Formal evaluation covers 15,600 episodes with zero training, fitting or optimizer updates. Evaluation JSONs total 2,230,267 bytes; remote pulls total 1,886,182 bytes. No checkpoints were downloaded; manuscript tables remain unchanged.

| Method | Radius 0.5 ID cost | Radius 0.5 50x cost | Radius 1.0 ID cost | Radius 1.0 50x cost |
| --- | ---: | ---: | ---: | ---: |
| Jacobian-only | 1.390151 | 4.727864 | 5.391373 | 8.679292 |
| No-symbolic | 2.117815 | 7.340855 | 8.607614 | 13.950246 |
| BAPR | 23.404715 | 237.017857 | 37.793037 | 260.236625 |
| RE-SAC | 15.505402 | 417.519862 | 22.204421 | 416.419702 |

Both expanded radii pass every frozen matched-seed learner-comparison gate at ID/50x. Radius-1.0 50x reductions versus no-symbolic/BAPR/RE-SAC are 37.78%/96.66%/97.92%. The own-fitted added-value gate fails separately at both radii: radius 0.5 loses on ID seed7 despite a 0.92% average 50x improvement; radius 1.0 is worse on both seeds at both endpoints, averaging 4.06%/2.12% higher ID/50x costs. Radius-1.0 50x own-fitted path wins are only 37/100 and 51/100; worst gaps are -4.109711/-1.842236 on path 60002. Full expansion confirmation is HOLD; the learner-baseline comparison passes.

## Limitations

These are the same two trained policies on already-used paths in one environment. Candidate action-component saturation fractions (`abs(u)>0.95`) are zero at both expanded radii, so action clipping does not explain the observed loss; action/Jacobian mismatch or a state-versus-effort tradeoff remains to be diagnosed. At radius 1.0, 50x observation-unit-ball exit fraction is 11.685%; this is a coverage diagnostic, not a derivative certificate. Own-fitted feedback and privileged known-model controls remain essential baselines; the latter are not optimality certificates.

## Next Step

Keep policies, references and environment frozen. Reuse inference JSONs for read-only action/output-force/Jacobian alignment and paired state-cost versus command-cost decomposition at radii 0.2/0.5/1.0, ID/50x, on all 100 paths, retaining path 60002. Diagnose why learning departs from its fitted reference before changing the environment, fitting, losses or seeds. No new tasks have been submitted.
