# v36: corrected reference-mapping evaluation

Frozen 2026-10-04 (Asia/Shanghai), after v35 training and before corrected
performance claims. v35 training remains unchanged. Its evaluator reconstructed
every fitted controller as `tanh(-K z)` from a gain, even for the cost-linear arm;
the actual arm reference is `clip(-K z, -1, 1)`. Therefore the v35 own-fitted gate
is invalid and must be recomputed on the completed remote checkpoints.

Run read-only evaluation only, using each completed v35 arm checkpoint on the
worker that owns it. The v35 shell command used `../checkpoints/seed{seed}` from
its own study directory. Therefore v36 must use the remote sibling path
`../../nonlinear_load_v35_reference_intervention_20261004/checkpoints/seed{seed}`
from the wrapper's `runtime/` working directory;
the v36 study's own `../checkpoints/` is not the source directory. The evaluator
must load the full checkpoint, assert the saved
reference arm and mapping, call the actual `LQRFeedbackReference` module for the
fitted controller, and perform zero optimizer updates. Use unchanged paths
66000--66099, radii 0.2/0.5/1.0, severity 1/50 and the same actor/known controls.
The actor returns must match the v35 actor returns exactly; otherwise stop.

Recompute only the own-fitted learner gate with the actual mapping. The v35
comparisons against the old cubic-local learner, radius-1 repair and small-radius
retention remain valid because they compare actor arms and do not depend on the
mislabelled fitted controller. No new seed, baseline, training, tuning or
checkpoint transfer to local storage is allowed. Collect compact JSON only.

If the corrected gate passes, admit at most the next prespecified learner
replication. If it fails, retain the OOD result as a bounded positive finding but
stop this reference branch; do not relabel the v35 outcome or expand experiments.
