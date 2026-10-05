# v35 evaluator erratum

The v35 learner training used the intended cost-linear `clip` reference. The
first formal evaluator, however, inherited the older gain-only helper and
reconstructed the fitted controller with `tanh` for both arms. Consequently the
reported v35 own-fitted gate was not a valid comparison for the cost-linear arm.

The actor-versus-old-arm results and the two repair/retention gates are unaffected.
The own-fitted decision is withdrawn pending the read-only v36 evaluation against
the actual saved reference mapping. No training result or checkpoint is changed.
