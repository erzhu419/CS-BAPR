# v35 status

2026-10-04 (Asia/Shanghai): reference-only learner comparison frozen;
[protocol](nonlinear_load_v35_reference_intervention_protocol.md).
50 focused/regression tests passed in 15.64s, covering clipped six-column
Jacobians, evaluation mapping, exact next-update restoration, shared full-state
prefix and unchanged individual-seed gates. Scheduler doctor passed.

Numerical preflight `t134705` completed with exit 0 on node007 GPU3. Its validator
passed: shared full-state prefix, restored reference mapping, samplers and cost
accounting; 1,400 actual training transitions / 401 updates, 11,000 shared
ID/validation transitions. Only 248,787 bytes of evaluation JSON were collected.
Separate preflight paths are not used for the formal performance decision.

Formal seed21 `t134718` and seed22 `t134719` completed with exit 0 under
automatic Linux GPU placement (`node007:GPU3` and `jtl110gpu:GPU0`). The formal
validator passed the protocol, full-prefix pairing and result accounting:
398,000 training transitions, 396,002 updates, 9,600 evaluation episodes and
6,812,482 bytes of compact evaluation JSON.

## Decision

The own-fitted numbers in this v35 report were provisional because the evaluator
reconstructed the fitted controller with a tanh mapping for both arms. They are
superseded for the cost-linear arm by the read-only corrected evaluation in
[`nonlinear_load_v36_reference_alignment_results.md`](nonlinear_load_v36_reference_alignment_results.md).
The v35 actor-versus-old-learner comparisons and actor-arm gates remain valid.

Against the old cubic-local learner, cost-linear improves all 12 individual
seed/radius/load blocks. The total-cost change is -1.31--3.96% under ID load
and -5.40--7.00% under 50x OOD load. The radius-1 repair and small-radius
retention gates pass. These are valid actor-arm comparisons. The corrected
own-fitted conclusion is reported in v36: the clip reference gate fails under
ID and passes under 50x OOD, so this is a positive OOD learner improvement
against the old learner, but not a qualified reference replacement for
replication.

The corrected evaluation shows the failure is localized to ID reference
tracking rather than OOD control:
at radius 1/ID the learner's excess is mostly state cost (+0.126/+0.170),
while command cost is slightly lower (-0.018/-0.007). The current actor remains
a smooth tanh policy while the confirmed reference is hard-clipped linear;
the next diagnostic should test this representation/loss-scale mismatch before
any new baseline or seed is added.

No learner replication is admitted. Historical failed gates and manuscript HOLD
remain unchanged. Checkpoints stay on workers; only compact JSON and logs were
collected.
