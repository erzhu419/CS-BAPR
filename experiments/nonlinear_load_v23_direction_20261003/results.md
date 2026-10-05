# Nonlinear load v23: critic directions, not an entropy-only explanation

Preflight t128875 and formal t128876 exited0; formal took116.1 seconds on automatically selected node007/GPU0. Both seed18 models shared that GPU and common paths/noise. All4,000 episodes passed validation; largest baseline error1.34e-6, zero updates/fitting, four focused tests passed. Only4,273,563 bytes of parameters/results returned; checkpoints/replay stayed remote.

## Results

Radius1 directional means toward the fitted latent mean, step0.005,100 paired paths and eight reparameterized draws. Lower cost and higher score are preferred.

| Model | Load | Discounted Cost | Q Mean | Soft Score |
| --- | --- | --- | --- | --- |
| floor | 1 | -0.151436 | -0.041583 | -0.054343 |
| no-floor | 1 | -0.154852 | -0.039615 | -0.045062 |
| floor | 50 | -0.150543 | +0.089660 | +0.107525 |
| no-floor | 50 | -0.150593 | -0.055215 | -0.252202 |

ID Q already opposes cost reduction; entropy removal did not resolve it. ID soft scores oppose51/79 and41/78 improving paths for floor/no-floor. Soft block-mean signs hold across all eight individual noise draws and deterministic zero-noise evaluation. Neither cost metric shows path-sign reversals across step0.01/0.005. Radius1 improvement comes from early regulation; radius0.2 improvement instead trades higher state cost for lower command cost.

## Next Step

Retain the floor/full Jacobian/cubic-local reference/environment and historical HOLD. Next compare frozen critics with paired stochastic continuations: common Gaussian draws, native tanh density, terminal reward normalization/alpha, gamma0.99 and native terminal handling. Separate physical return, future entropy and initial entropy before choosing a critic/target intervention. No risk-weight sweep, seed expansion, follow-on training or manuscript change.

## Limitations

Deterministic costs differ from stochastic soft targets: this is not a critic-error certificate. Local output-mean derivatives do not reconstruct full training gradients or establish general RL superiority.

Evidence: `results/nonlinear_load_v23_direction_diagnostic/summary.json` and the frozen manifest; reproduce with `scripts/summarize_nonlinear_load_v23.py` under the frozen runtime.
