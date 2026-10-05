# LQR feedback_v2 results

All 15 manifest tasks completed with exit code 0. Every evaluation JSON contains 500 training episodes, 100,000 environment steps, 99,001 updates, and 20 paired evaluation seeds at each of six amplitudes. The 15 source JSON files total 310,910 bytes; no remote checkpoint or CSV was downloaded.

Mean episodic return, in thousands, averaged over three training seeds; higher is better. Training-seed sample standard deviations and individual seed means are retained in `results/lqr_feedback_v2/summary.json` and `seed_mean_returns.csv`.

| Method | 1x | 2x | 5x | 10x | 20x | 50x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CS-BAPR | -0.661 | -17.906 | -577.268 | -1108.241 | -2328.985 | -7921.523 |
| No symbolic alignment | -145.076 | -174.811 | -257.008 | -1214.150 | -2671.599 | -9267.178 |
| No arithmetic actor | -31.911 | -65.467 | -138.230 | -1537.158 | -3032.049 | -9535.515 |
| BAPR | -26.435 | -26.411 | -278.704 | -1732.482 | -3459.538 | -10850.976 |
| RE-SAC | -0.856 | -1.465 | -216.042 | -1855.058 | -4006.432 | -11302.709 |

## Findings

- At 50x, CS-BAPR reduces mean cost by 27.0% versus BAPR and 29.9% versus RE-SAC, winning all three matched training-seed comparisons against both. At 20x, the reductions are 32.7% and 41.9%; wins are 3/3 and 2/3, respectively.
- At 5x, CS-BAPR is worse than both comparators on average: costs are 577,268 versus 278,704 and 216,042. Removing the arithmetic actor gives a mean cost of 138,230. There is no uniform advantage across amplitudes.
- SINDy identification R-squared exceeds 0.999999999998 in every symbolic run. The full actor's curvature bounds range from 299,474 to 1,142,049, versus 5,430 to 8,581 for the smooth MLP symbolic arm. Identification accuracy and a finite certificate have not yielded a small certificate or reliable policy extrapolation.
- A post-hoc diagnostic using the same evaluation states gives zero-action costs of 1,018,450 at 20x and 6,365,310 at 50x, below CS-BAPR's 2,328,985 and 7,921,523. Tanh feedback from the known dynamics gives 20,971 and 1,234,719. This environment admits substantially better control; merely beating the two learned comparators is insufficient.

## Limitations

This is initial-state extrapolation under fixed linear dynamics, not bus demand randomness or general nonstationary RL. Three training seeds do not establish statistical significance, and seeds ran on different GPU/PyTorch combinations. Both comparators inherit shared project stabilization. The known-model feedback diagnostic was added after the frozen study, uses privileged dynamics, and is not the constrained finite-horizon optimum or a replacement primary comparator. Historical manuscript evidence remains separate.

## Next Step

The read-only checkpoint analysis is complete in `docs/lqr_feedback_v2_diagnostics.md`. It finds correct fitted feedback gains, substantial action bias, weak Jacobian gradients, and a shared critic target-scale mismatch. Prioritize an isolated action-value-anchor ablation; keep this completed study unchanged.

Reproduce the aggregation and control diagnostic with `experiments/lqr_feedback_v2_20260930/runtime_r2/.runtime/bin/python scripts/summarize_lqr_feedback_v2.py`.
