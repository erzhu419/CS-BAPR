# v77 Mean/Covariance Attribution

Replay all 65 saved v76 evaluation paths as development evidence. Cross the
frozen/updating mixed banks' conditional residual means and covariances in a
2x2 comparison. Preserve forward coefficients, input/target scales, 64-neighbor
rules, recursive hazard 0.02 and episode handling. Each covariance remains
conditioned on its own bank's input neighborhood; no moments are refitted.

Reuse the two diagonal references and require their scalar window statistics
to reproduce v76. Calibrate only the two hybrids' normal risk thresholds, on
the same saved v69 normal-null population. No test labels enter calibration.

Report normal risk, early/tail accuracy, mode margins and paired risk contrasts
under the existing screen rule. For first_post50/last64, decompose true-versus-
best-other and every fixed competitor log-density margin into error energy
and log determinant. Compare conditional prediction RMS error with the nearest
other saved counterfactual observation template gap in common target units.
Historical v61/v62 paths retain all density/belief diagnostics; only template-
gap metrics are unavailable because those paths did not save counterfactuals.

No simulation, optimizer updates, seed expansion or actor feedback. Synthetic
tests and a native saved-bank density-equivalence smoke gate precede dispatch.
Resume uses scalar completed-path records and saved hybrid threshold curves;
source references/traces remain read-only. Only scalar JSON/logs return locally.
Any selected change subsequently requires fresh confirmation.
