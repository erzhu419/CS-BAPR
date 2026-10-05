# Nonlinear load v22: no stable benefit from removing the entropy floor

All four tasks t128677--t128680 exited0 in50--82 minutes. The four evaluation JSONs were already local (6,871,654 bytes); no remote data/checkpoint pull was needed. Frozen validation passed400,000 learner interactions,396,004 updates,40,000 identification plus4,000 validation interactions,9,600 controller episodes, runtime, temperature records, paired fitted/known controls and cost decompositions.

## Results

Radius1 actor cost change after disabling the floor, relative to the same-seed floor arm. Negative is better; each block has100 paired paths.

| Seed | ID | 50x Load | Pre-Floor Episode Traces |
| --- | --- | --- | --- |
| 17 | -0.696% | -0.570% | Diverged before the intervention |
| 18 | +0.367% | +0.134% | Matched through episode85 |

**HOLD.** Radius1 total/early-state improvement, zero-margin radius0.2 retention and all-radius own-fitted added value all failed. The no-floor actor beats its own fitted reference only at seed17/radius0.2/50x. Seed18/radius0.2/ID cost rises0.0194%. At radius1, seed18 early-state cost increases0.04397/0.04294 at ID/50x; lower command cost does not offset this regulation loss.

The switch was effective: terminal alpha falls from0.01 to0.001029/0.000878 for seed17/18. Floor-arm constraints bind on83,087/82,904 of99,001 updates; terminal reward standard deviations remain about13.7--14.6. Failure is not an inactive-switch artifact.

## Pairing Issue

Seed17 uses RTX2080Ti versus RTX3080Ti. Before either arm crosses the nominal floor, recorded returns differ from episode6 in79 of84 episodes (maximum absolute difference569.28); alpha and reward-scale traces also differ. Five matching exploration episodes and matching identification therefore do not establish a common learned trajectory. Its small final improvements are descriptive, not an isolated causal estimate of floor removal. Seed18's returns/alpha/reward-scale traces match exactly for the first85 episodes on RTX3080Ti workers, yet radius1 worsens after the intervention. The result does not support adopting the no-floor variant.

## Next Step

Keep the default floor, full Jacobian and cubic-local reference; do not expand the no-floor variant or retune its seeds/thresholds. Future counterfactual training should run both arms on one scheduler-selected Linux GPU from a shared complete prefix, preserving free placement without node preferences. Use the matched seed18 cohort for read-only actor/critic diagnosis before another intervention. No new tasks were submitted; historical HOLD decisions and manuscript tables remain unchanged.

The read-only step is now completed as v23. Reparameterized ID Q directions disagree with deterministic cost-reducing directions even without the entropy floor; stochastic continuation alignment is the next diagnostic, not another entropy sweep. See `docs/nonlinear_load_v23_direction_results.md`.

## Limitations

Matching episode-level traces does not certify identical intermediate parameters. Cross-device divergence is observed, but its precise numerical cause is not established. Two seeds and reused evaluation paths in one fully observed environment do not establish general RL or learner-comparator superiority; fitted references are feasible controls, not certified optima.

Evidence: `results/nonlinear_load_v22_entropy_floor/{task_observations,summary}.json`, four per-arm/per-seed evaluation JSONs, and the frozen study manifest. Reproduce with `python3 scripts/summarize_nonlinear_load_v22.py` using the frozen package versions. Prefix diagnostics were added after observing the mismatch; original admission gates were unchanged, and two focused prefix tests passed.
