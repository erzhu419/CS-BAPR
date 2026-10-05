# v36 corrected result

The v35 evaluator bug has been isolated: the cost-linear arm was evaluated with
`tanh(-Kz)` instead of its saved `clip(-Kz,-1,1)` reference. v35 training and
actor-versus-old-arm results are unchanged, but its own-fitted learner gate is
withdrawn until this read-only correction is complete.

The v36 evaluator loaded the completed remote checkpoints from the v35 sibling
study directory, performed no training or optimizer update, asserted the saved
mapping, and reused paths 66000--66099. Both formal tasks exited 0: `t135188`
on node007 for seed21 and `t135189` on jtl110gpu for seed22. The four compact
JSON files total 6,199,110 bytes.

The actor returns reproduce v35 exactly for all 24 radius/severity/seed blocks.
With the actual saved mappings (tanh for cubic-local and clip for cost-linear),
the corrected cost-linear own-fitted excess is:

| Radius | ID seed21 | ID seed22 | 50x OOD seed21 | 50x OOD seed22 |
|---|---:|---:|---:|---:|
| 0.2 | +0.000176 | +0.000120 | -0.951837 | -1.415367 |
| 0.5 | +0.009732 | +0.015003 | -0.913708 | -1.353982 |
| 1.0 | +0.108210 | +0.162411 | -0.820827 | -1.223681 |

Therefore the corrected own-fitted gate fails under ID for all six blocks and
passes under 50x OOD for all six blocks. The radius-1 repair and small-radius
retention gates remain passed, but learner replication is not admitted. This
confirms a bounded OOD improvement of the cost-linear learner over the old
cubic-local learner, not a general reference replacement. No new seed,
baseline, training, tuning, or checkpoint transfer was performed.

## Cost decomposition diagnostic

The ID failure is not uniform at the smallest radius. At radius 0.2, seed21
has positive state-cost excess and negative command-cost excess, whereas seed22
has the opposite pattern; the small positive total excess is therefore not
evidence for one deterministic failure mechanism. At radii 0.5 and 1.0, both
seeds have positive state-cost excess, and at radius 1.0 it dominates the
negative command-cost excess (`+0.1258 - 0.0175 = +0.1082` for seed21 and
`+0.1698 - 0.0074 = +0.1624` for seed22).

Under 50x OOD, the improvement is mainly command-cost reduction: the command
component is negative in all six blocks and accounts for most of the total
excess. This supports an ID representation/reference-tracking diagnostic as
the next algorithmic step, but does not justify adding seeds or claiming a
general-purpose adaptation mechanism.
