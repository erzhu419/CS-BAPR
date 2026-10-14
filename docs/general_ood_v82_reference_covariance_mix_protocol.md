# v82: Reference-selected covariance mixture

Keep the v78 local-linear residual mean, 64 neighbors, unpenalized intercept and Ridge alpha 0.064. Mix the regularized updating-bank raw-residual OAS covariance (v78 endpoint) with the zero-centered local LOO prediction-error OAS second moment (v81 endpoint). One global raw-covariance weight is shared by all modes and queries.

Recover the saved 23,000-row updating bank's provenance from server-side reference traces. Hold out each of six reference seeds (135050, 135090, 135091, 135092, 135170, 135171) in turn. Every coupled physical stream and all five counterfactual labels of the held-out seed leave both the local mean fit and both covariance neighborhoods. Forward coefficients and input/target scaling stay frozen from their separate fit population.

Select exactly 512 evenly spaced bank rows per held-out seed, retaining all five known-mode labels: 3,072 validation queries and 15,360 residual labels. Minimize equal-seed, equal-mode Gaussian negative log likelihood over the raw weight in [0,1], using the existing bounded optimizer and explicitly comparing both endpoints. Omitted density constants are independent of this weight. Freeze selection before loading development outcomes or normal-null traces. Restore the full reference bank for every subsequent readout.

Native verification checks exact reference-row alignment, v78 mean/error invariance and both density endpoints. If selection returns a pure endpoint, report that outcome without dispatching a duplicate replay. Otherwise recalibrate only the departure curve on the same 12,000 normal-null transitions and evaluate the same 65 six-thousand-step development paths with unchanged recursive rules and windows. Retain all five v78 controls and the v81 endpoint as reused statistics, not new replications.

Evaluate early/tail mode accuracy, signed recursive true-mode margins, normal risk and change-minus-no-change contrasts. A gain in accuracy does not override newly overconfident wrong-mode margins. This round generates no environment transitions or actor updates and does not change paper tables.

## Limitations
Known-mode, reference-seed validation cannot establish arbitrary OOD extrapolation. These already-inspected development paths and one source training seed cannot establish general RL return improvements. Full arrays, reference banks and caches remain on the server; only scalar JSON and logs are synchronized locally.
