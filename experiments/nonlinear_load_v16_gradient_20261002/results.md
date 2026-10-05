# Nonlinear load v16: terminal gradients do not explain v15 degradation

CUDA preflight t125003 and formal tasks t125006/t125007 exited 0 on scheduler-selected jtl110gpu2 GPUs, without placement preferences. Formal diagnostics took 20.7/21.8 seconds. All four existing candidates reproduced their 2,400 paired actor evaluation episodes; maximum return error was 6.8e-6. Native Jacobian checks, finite gradients and unchanged actor/critic parameters passed. No training, fitting or optimizer updates. Three focused tests passed locally and in the frozen runtime.

Four compact parameter/sample JSON exports totaled 3,182,478 bytes. The full checkpoints and replay stayed remote. Results and frozen inputs are under results/nonlinear_load_v16_gradient_diagnostic and experiments/nonlinear_load_v16_gradient_20261002.

## Findings

For both balanced policies, all eight noise draws show conflicting SAC/Jacobian gradients on recent replay. Nevertheless, their combined replay descent decreases initial physical-Jacobian mismatch at radius1 under both loads. This also holds for both narrow policies, and for all-replay mean gradients. A persistent terminal combined-gradient mechanism pushing initial mismatch upward is not supported.

| Balanced policy, recent replay to radius1/ID | SAC descent derivative | Jacobian descent derivative | Combined derivative |
| --- | ---: | ---: | ---: |
| seed9 | +4.5644e-5 | -9.8336e-5 | -5.2692e-5 |
| seed10 | +8.7824e-5 | -1.1908e-4 | -3.1259e-5 |

These are derivatives of the 0.01-weighted initial physical-input Jacobian loss under the mean negative parameter-gradient direction, not predicted optimizer steps. Positive means worsening. Recent replay Jacobian-gradient norms are about 55% of SAC norms in both balanced policies: the constraint is active and offsets the SAC contribution. This does not justify increasing its global weight.

In the final 100 training episodes, physical-state norms >=0.5 occur in 2.815%/3.310% of balanced replay, versus 0.265%/0.280% of narrow replay. Resets themselves occupy 0.5% of stored transitions. Wider reset support exists but remains weakly represented in the transition-weighted objective.

## Next Step

Test one sampling intervention: allocate half the Jacobian state batch to stored reset observations and half to ordinary replay, using an independent sampler. Keep the SAC batch, 0.01 Jacobian weight, actor, critic, fitted reference, balanced reset cycle and interaction/update budgets unchanged. Compare against unchanged sampling on two fresh matched seeds before expanding learner comparisons. Require lower radius1 total and early-state cost on both seeds at ID/50x, while retaining radius0.2 load performance and the existing own-fitted added-value criterion. No v17 training tasks have been submitted; historical v13/v15 HOLD remains unchanged.

## Limitations

Terminal frozen gradients do not reconstruct historical optimization or Adam moment/weight-decay effects. The support statistics are norm marginals, and OOD-state critic gradients can be inaccurate. Sampling bias remains a testable hypothesis rather than an established cause of the v15 loss.
