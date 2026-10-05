# v31 results

2026-10-04 (Asia/Shanghai): both paired tasks complete; formal validation passed.
Protocol: [v31](nonlinear_load_v31_symbolic_protocol.md).

The v30 environment qualified; this round measures symbolic Jacobian added value
with fresh seed21/22, weight 0 versus 0.01, 500 episodes per arm, early critic
and uniform actor. Historical HOLD and manuscript remain unchanged.
Local verification: 41 tests passed in 13.76s; launcher syntax and
project-scoped scheduler doctor passed. Wrong loss weight, environment metadata
and missing control were rejected; the formal validator's old/default and fresh
path interfaces passed a synthetic metadata transformation check.

Preflight `t132397`: node007 GPU3, exit 0, 54.72s scheduler lifetime; 1,400 actual
training transitions and 401 updates. Shared prefix/reference, controls,
sampler states/counts, runtime and finite cost accounting passed. Pulled two
compact evaluation JSON files, 248,163 bytes total; no checkpoint/replay pull.
This is numerical admission only, not performance evidence.

Formal `t132398`/`t132399` exited 0 on node007 GPU3/GPU0 in 100.37/110.21 minutes.
All four runs and 9,600 evaluation episodes passed the frozen protocol, including
shared prefix/reference, sampler states, controls and cost accounting. Only
6,769,714 bytes of formal evaluation JSON were synchronized; no checkpoint pull.

| Radius 1 | Seed | Off Cost | On Cost | On vs Off |
| --- | ---: | ---: | ---: | ---: |
| ID | 21 | 10.163805 | 5.294374 | -47.91% |
| ID | 22 | 6.213317 | 5.246032 | -15.57% |
| 50x OOD | 21 | 237.950715 | 54.249038 | -77.20% |
| 50x OOD | 22 | 294.283620 | 54.196945 | -81.58% |

On reduces total cost in all 12 blocks; OOD reductions are 77.20--82.76%.
Radius-1 repair and small-radius retention pass. Own-fitted added value fails:
all six OOD blocks improve by 0.40--0.66%, but all six ID blocks are worse by
0.009--1.36%. The ID physical cubic-fit RMSE is approximately 1e-7. The reference
mapping remains local-linear despite identifying the nonlinear actuator.

The [v32 reference-only diagnostic](nonlinear_load_v32_reference_results.md) is
also complete: the ID-fitted actuator inverse improves OOD but worsens ID, so
direct reference replacement is not admitted. Next is a separate diagnostic of
an ID-cost-aware nonlinear reference. No seed/baseline expansion or learner
replication admitted.

## Limitations

The large on/off gap establishes added value of this constraint within the two
seed sustained-load ablation, not general RL superiority. On remains approximately
10% worse than known-inverse under OOD, and nearly matches its simple fitted
reference. The failed own-fitted gate and all historical negative evidence remain.
Evidence: `results/nonlinear_load_v31_symbolic/{summary,block_analysis,task_observations}.json`.
