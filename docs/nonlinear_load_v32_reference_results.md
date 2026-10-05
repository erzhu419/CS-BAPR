# v32 status

2026-10-04 (Asia/Shanghai): both formal tasks complete; validation passed.
Protocol: [v32](nonlinear_load_v32_reference_protocol.md). Existing ID fits and
seed21/22 are reused; no fitting, learner updates or checkpoint access.

The local/regression suite passed 35 tests; launcher syntax and project-scoped
scheduler doctor passed. Preflight `t134561` exited 0: 60 episodes, 12,000
transitions and 53,337 bytes of evaluation JSON. Source controls reproduced and
the origin action Jacobian was preserved. Input summaries total 20,647 bytes.

Formal CPU tasks `t134571` (seed21) and `t134572` (seed22) both exited 0 on the
automatically selected local WSL node, about 13.71s each. Three controllers,
three radii, ID/50x OOD and 100 fresh paths per block: 3,648 episodes including
source-control reproduction, 729,600 transitions. Validation passed; only
2,100,702 bytes of local evaluation JSON were collected, with no remote pull.

Each seed has the same directional result; costs below are seed21, and the
relative changes agree with seed22 within 0.000005 percentage points.

| Radius | ID Local | ID Inverse | ID Change | OOD Local | OOD Inverse | OOD Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.262596 | 0.263037 | +0.168% | 50.614043 | 46.298587 | -8.526% |
| 0.5 | 1.348856 | 1.377292 | +2.108% | 51.421943 | 47.224766 | -8.162% |
| 1.0 | 4.929485 | 5.186387 | +5.212% | 54.609152 | 50.774186 | -7.023% |

All six OOD blocks improve; all six ID blocks worsen. OOD path wins are
96/95/91 out of 100 at radii 0.2/0.5/1.0; ID path wins are only 29/3/2.
The frozen all-block gate fails, so reference replacement and learner
replication are not admitted.

## Interpretation

The new reference lowers command cost in every block, but raises total state
cost in every block. At radius 1, seed21: ID state cost increases 0.296895 while
command cost decreases only 0.039993; OOD state cost increases 0.683937 while
command cost decreases 4.518903. ID degradation is concentrated in the first
50-step window. Initial load-cancellation Jacobian residuals decrease, but that
does not imply lower actual state/command cost.

This points to the control objective, not additional identification data: an
actuator inverse with the fixed local gain is insufficient. Next is a separate,
reference-only diagnostic using the ID nonlinear dynamics and the unchanged
state/command cost to determine control actions. Retain the old local and v32
inverse references as controls, use fresh paths, and require strict total-cost
improvement in every seed/radius/load block before a learner intervention.
The separate [v33 cost-aware diagnostic](nonlinear_load_v33_cost_reference_results.md)
is complete: its nonlinear candidate improves ID but worsens OOD, failing both
formal gates. v32's failed reference replacement remains unchanged.

Evidence: `results/nonlinear_load_v32_reference/{summary,block_analysis,task_observations}.json`.

## Limitations

This diagnoses references, not trained learners. The two ID fits are nearly
identical; matching outcomes are not two independent learner replications.
Historical HOLD, the failed v31 own-fitted gate and the manuscript remain unchanged.
Two startup-only attempts failed before numerical output; the launcher runtime
path and stale staging were repaired. Their receipts remain in the manifest.
