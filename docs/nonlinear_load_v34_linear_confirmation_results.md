# v34 independent confirmation

2026-10-04 (Asia/Shanghai): formal seed21 `t134595` and seed22 `t134591` completed
with exit 0. [Protocol](nonlinear_load_v34_linear_confirmation_protocol.md).
All four source controls and the unchanged v33 cost-linear gain reproduced.
The formal validator passed: 4,896 episodes including reproduction, 979,200
transitions, and 2,918,488 bytes of evaluation JSON. Fresh paths 65000--65099
are separate from the numerical preflight and v33 paths.

Cost-linear improves mean total cost against fitted-local in all 12 primary
seed/radius/load blocks. Across individual blocks, ID cost decreases
1.13--4.23%, and 50x OOD cost decreases 4.22--4.37%; no seeds are pooled.
For seed21 (seed22 has the same directions and rounded percentages):

| Radius | ID Cost Change | 50x OOD Cost Change |
| --- | ---: | ---: |
| 0.2 | -1.132% | -4.318% |
| 0.5 | -2.266% | -4.368% |
| 1.0 | -4.228% | -4.220% |

The gate admits one reference-only learner intervention. Next, preserve the
unchanged clip mapping through training, checkpoint restoration and evaluation,
including Jacobian and restoration tests, before freezing that comparison.
The [v35 reference-only learner trial](nonlinear_load_v35_reference_intervention_protocol.md)
passed numerical preflight; formal tasks `t134718`/`t134719` were submitted.
Learner outcomes remain pending, not evidence supplied by this confirmation.

Verification: 39 focused/regression tests passed in 6.35s; launcher syntax and
project scheduler doctor passed. Numerical preflight `t134589` exited 0 and
validated 19,200 transitions; its paths do not enter the formal decision.
Evidence: `results/nonlinear_load_v34_linear_confirmation/{summary,block_analysis,task_observations}.json`.

## Limitations

This confirms a reference, not actor learning or independent learner replication.
Cost-linear remains 3.28--5.12% worse than fitted-inverse under OOD. Historical
HOLD and earlier failed gates remain unchanged; there was no new identification,
training, optimizer update or checkpoint access. Existing learner helpers still
reconstruct tanh references and must be changed before a clip-reference trial.
The original seed21 task `t134590` failed before Python started because staging
included a local runtime marker; the environment was isolated and `.runtime/`
excluded before the unchanged replacement run. No numerical outcomes were discarded.
