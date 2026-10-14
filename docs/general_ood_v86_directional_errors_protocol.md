# v86 Directional error diagnosis

Freeze v78 means, v78 raw covariance and the v83-selected covariance mixture. Inspect all 65 existing paths at first_post50/last64: 130 windows / 7,410 transitions, including failed, improved and unchanged controls. No new trajectory or readout replay is generated.

For true mean mu_t, competing mean mu_c and observation y, define e = y - mu_t, d = mu_t - mu_c, and P_t/P_c as inverse covariances. The fixed-pair log-density margin decomposes exactly into e' P_t d + 0.5 d' P_t d (common-covariance mean position), 0.5 (e+d)' (P_c-P_t) (e+d) (covariance shape), and 0.5 log(det C_c / det C_t) (volume).

Project query and the same 64 neighbors' frozen whole-seed-held-out errors along -P_t d / sqrt(d' P_t d); positive means toward the competing predicted mean. Compare signed means, second moments, reference 95th-quantile exceedances and common-covariance halfway-boundary crossings. Equal predicted means have no separation projection. When saved counterfactual observations exist, compare predicted and actual mode-separation directions in the same metric.

Local tests cover Gaussian/decomposition correspondence, boundary sign, volume reversals, unit invariance and source mismatch rejection. The server task must reproduce v78/v83 window RMS, likelihood, energy/volume and all fixed-pair comparisons before interpretation. A mismatch stops the diagnosis for correction. Only scalar JSON/logs return; banks/traces/errors remain on node007 with the native runtime. No weights, actor, environment or paper tables change.

## Limitations
These are development diagnostics, not independent confirmation or RL return evidence. Reference errors use smaller fitting banks and are transported into query directions; quantiles are not confidence intervals. Algebraic components do not by themselves identify a unique causal fix.
