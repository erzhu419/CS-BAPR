# v42 Frozen Dynamics Results

2026-10-06: preflight t135901 and formal t135902 completed on node007 GPU0, exit 0. Formal: 200 rollouts, 307200 interactions, 312.70 seconds, zero RL updates/replay writes/new training seeds. Peak VRAM 234 MB, RAM 721 MB. Only 2285547 bytes of formal JSON synced; checkpoints stayed remote. Nine focused tests passed. Config/runtime, paired prefixes and all reward/ending traces match v40/v41 sources.

## Findings

Quadratic state-action residuals on the adaptive policy increase versus no-change in all 20 pairs for each of the four modes, during the first 16 transitions, with no physical endings there. This recovers a weak early physics response missed by reward reference. However, all 1280 changed-mode early transitions remain below their calibration95 threshold. First-post mean scores are 0.01441 heavy, 0.00008366 slippery, 0.00005499 high_grav, 0.0002090 compound versus no-change 0.000002025. Relative increases alone do not establish usable calibration.

The same predictor on fixed-minus2 has no-change calibration95 exceedance rates 26.74% in held-out validation, 36.91% in first-post256 and 43.09% in second-post256, rather than a stable approximately 5% reference. Heavy early mean contrast is +0.3514 but positive in only 12/20 pairs; high_grav is -0.1493, positive in 5/20. Slippery/compound have stronger early positive responses in 20/20. Linear and state-only controls also lack a uniformly reliable signal. State-only residuals reproduce adaptive early positive contrasts in all modes, so action conditioning has not been established as the source of the response.

Forecast error changes sharply between fitting and calibration. Mean quadratic state-action MSE in fitting-target units rises from 0.0004152 to 0.9600 for fixed-minus2, and 0.0002379 to 307.54 for adaptive. Adaptive controls physically terminate in validation in all 20 probes; their post-reset no-change scores become extremely small. A single calibration scale does not represent both trajectory regimes. Limited fitting coverage and regime mismatch are the leading issues to test next.

Explicit flags resolve the v41 ambiguity: fixed-minus2 has no physical terminations in any case; its 20 endings per profile in the last window are all TimeLimit truncations. Adaptive has no truncations, with heavy physical terminations in first-post256 and high_grav/compound in second-post256. Both post windows remain reported.

Judgment: HOLD for belief-feedback training. Early residual differences are real paired observations, but poor null forecasting and regime-dependent scales make this calibration unsuitable as a drop-in surprise signal.

## Next Step

Collect independent normal-only reference trajectories spanning complete episodes and resets from both frozen policies. Use disjoint trajectories for fitting, calibration and testing. First keep predictor capacity fixed to determine whether broader coverage repairs null forecasting; then test centered vector residuals and state/episode-conditioned scales. No OOD-based threshold tuning, RL seed expansion or NAU/SINDy restoration. Advance to matched-budget belief feedback only after useful null calibration and early change response are established together.

## Limitations

One training seed, reused development evaluation seeds, deterministic policies and shared no-change controls. Polynomial capacity, partial observation/clipping and extrapolation may contribute; this run does not isolate their causal contributions. Calibration95 is an empirical reference, not a guaranteed false-alarm level. This is not an extreme50x benchmark, policy-return improvement or independent confirmation.

Summary: `results/general_ood_v42_frozen_dynamics/summary.json`.

Follow-up v43 preserves predictor capacities, adds independent full-episode normal references and fixed calibration controls, and exactly reproduces v42 test trajectories. It substantially improves early detection and adaptive null calibration; see `docs/general_ood_v43_normal_reference_results.md`. The original v42 artifacts and HOLD judgment remain unchanged.
