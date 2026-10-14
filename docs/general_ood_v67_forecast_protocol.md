# v67 causal residual forecast

Keep the v66 level channel exactly fixed, including its normal-only covariance
weight, score scale and age/phase thresholds. Add a normal residual forecaster:
current11 observations + three executed actions + previous16 residual mean
(11 coordinates) + prior history fraction. The26 inputs precede the current
next-state observation. Fit a standardized linear ridge predictor of the
current normalized dynamics residual, using existing mean-loss ridge1e-3 and
an unpenalized intercept. No current residual enters its own prediction.

Fit using the two existing v62 normal-fit traces (seed134000, frozen/updating,
12000 transitions). Fit center and OAS covariance of the causal16-step mean of
forecast errors, with existing relative ridge1e-3. Separate seed134010 normal
calibration traces (12000 transitions) set score95 and the existing phase/age95
departure thresholds. No OOD fitting, window/threshold search or new RL budget.

Compare forecast-only ablation with the predeclared main risk union:
`max(v66 mean risk, forecast mean risk)`, each using its unchanged16-step risk
history. The old v66 channel is not jointly recalibrated. Residual, input,
forecast-error and risk histories persist across observed episode resets.
Known modes/switch tick are offline labels only.

Replay all17 existing v62/v61 trajectories and inherit eight controls unchanged.
Reconstruct the v66 level from each raw server trace to check risk/signal means.
Keep groups separate and all frozen/updating normal first800/full/last1000/last64
mean risks <=0.10. Every changed first50/tail64 paired contrast must remain >0.
Only a main-candidate pass in both groups warrants new continuations.
CPU replay on node007; return scalar JSON/logs, keep references/raw data there.

## Limitations

The risk union preserves a per-transition v66 floor, not normal bounds or
paired contrasts. All trajectories are reused development data from one source
training seed; neither the forecast fit nor fixed-beta replay establishes
independent confirmation or adaptive RL gains.
