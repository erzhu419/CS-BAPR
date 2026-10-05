# v33 results

2026-10-04 (Asia/Shanghai): cost-aware reference diagnostic frozen; 31 focused
and regression tests passed in 6.44s. Launcher syntax and project scheduler
doctor passed. [Protocol](nonlinear_load_v33_cost_reference_protocol.md).
Preflight `t134575` exited 0 on local WSL in 46.93s scheduler lifetime; 96 episodes,
19,200 transitions, all old controls reproduced and solver accounting validated.
Only 79,755 bytes of local result JSON collected. Input JSON totals 18,965 bytes;
pinned v32 dependencies reused without additional local package storage.

Formal CPU tasks `t134576`/`t134577` exited 0 on automatically selected local
WSL, about 24.69s each. Validation passed: five references, three radii, ID/50x
OOD, 100 new paths per block, 6,072 episodes including reproduction and 1,214,400
transitions. Collected 3,502,993 bytes of local JSON, with no remote transfer.

Both seeds show the same directions; seed21 relative cost changes versus old
fitted-local are below. Every individual block is retained in `block_analysis.json`.

| Radius | ID Nonlinear | OOD Nonlinear | ID Linear | OOD Linear |
| --- | ---: | ---: | ---: | ---: |
| 0.2 | -1.374% | +18.709% | -1.230% | -4.077% |
| 0.5 | -5.874% | +18.227% | -1.962% | -4.135% |
| 1.0 | -7.095% | +16.852% | -4.660% | -4.072% |

The nonlinear candidate fails both all-block gates; no learner intervention is
admitted. Its OOD saturation fraction is 19.43--19.87%, versus 2.46--3.24% for
cost-linear. Both state and command costs increase under OOD, across all four
windows; it wins zero of 100 OOD paths against cost-linear in each block.
The solver completed 480,000 component minimizations with maximum objective
excess versus cost-linear below 5e-25. The evidence is consistent with a
local-quadratic terminal-value mismatch, rather than failure to minimize the
specified one-step objective.

The positive cost-linear secondary control was separately frozen and tested in
[v34 confirmation](nonlinear_load_v34_linear_confirmation_results.md), retaining
seed21/22 ID fits and the strict all-block improvement criterion on fresh paths.
`t134595`/`t134591` completed with exit 0 and all 12 v34 primary blocks improved,
admitting one reference-only learner intervention. This confirms a hypothesis
from a secondary control, not a passed v33 primary gate; the failed v33 primary
decision remains unchanged.

Evidence: `results/nonlinear_load_v33_cost_reference/{summary,block_analysis,task_observations}.json`.

## Limitations

This evaluates reference controllers, not learners. Cost-linear is still worse
than v32 fitted-inverse in all OOD blocks. Its positive secondary result was
independently confirmed in v34, not as learner evidence. No RL training, new identification or checkpoint access;
historical HOLD, prior failed gates and the manuscript remain unchanged.
