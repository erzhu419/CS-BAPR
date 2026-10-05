# Nonlinear load v14: radius mechanism diagnosis

Tasks t122953/t122954 (seed7/8) ran on scheduler-selected node007 GPU1/GPU2 and exited 0 in about 40/33 seconds. All 4,800 controller episodes reproduced v13 within the frozen tolerance; the maximum absolute return deviation was 1.31e-6. State/command accounting, phase splits and paired per-path excesses were independently checked; nine focused tests passed. Only 3,267,942 bytes of formal diagnostic JSONs were pulled, with no checkpoints, training, fitting or manuscript changes.

| Reset radius, 50x load | First 50 steps cost excess | Last 150 steps cost excess | Total cost excess |
| --- | ---: | ---: | ---: |
| 0.2 | -0.020815 | -0.029068 | -0.049883 |
| 0.5 | -0.014718 | -0.029180 | -0.043898 |
| 1.0 | +0.208614 | -0.028690 | +0.179924 |

Values are mean actor-minus-own-fitted costs across the two training seeds; positive is worse. At radius 1.0/50x, state cost rises 0.367983 while command cost falls 0.188059. Both seeds retain a late-phase improvement but lose more in initial regulation. Initial-state physical-input action-Jacobian squared mismatch rises from 0.015366 to 0.202378 (13.17x); ID shows the same pattern (13.14x). Path 60002's excesses remain +4.109711/+1.842237, concentrated in early state cost. The immediate bottleneck is large-initial-state regulation rather than loss of the observed late-phase load benefit.

## Limitations

This is retrospective diagnosis of the same two actors, same paths and one environment, not an independent confirmation or proof that insufficient training-state coverage is the sole cause. v13's failed own-fitted expansion gate is unchanged. Larger radii included in future training would no longer be unseen-reset OOD; that experiment would test load robustness across covered state scales, not repair the original unseen-state claim.

## Next Step

Keep the algorithm, symbolic-identification settings, load process and interaction budgets fixed. Change only RL training reset coverage to a balanced 0.2/0.5/1.0 schedule, shared by all four learned methods. Retain separate ID/50x evaluation at each radius and own-fitted feedback controls; use fresh matched training seeds. Test whether this removes the early state-cost penalty before making further loss or environment changes. No additional training tasks have been submitted.
