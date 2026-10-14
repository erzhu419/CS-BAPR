# v44: Signal response and residual-level adaptation

Task `t135911` completed on node007 GPU0 in 382.93 seconds. Two frozen seed1
policies, 24 normal reference rollouts and 200 paired test rollouts: 380,928
interactions, zero RL updates/replay writes. Only 7,081,635 bytes of policy JSON
were synced, no checkpoints/trajectories. All selected v43 model metadata,
residual summaries, configs/runtime and reward/ending traces match; 14 tests pass.

Adaptive Student-t early16 lambda exceeds matched sham in all 160 changed
cases; no first16 physical failures or sham majority resets (0/40).

| Frozen policy | Early lambda: normal / changed | Late64 changed mean age | Normal validation peak reset > .5 |
|---|---:|---:|---:|
| fixed-minus2 | .157 / .524 | 19.00 | 15/20 |
| relative-adaptive | .176 / .498-.552 | 16.50-18.97 | 5/20 |

On fixed-minus2, late changed log-signal remains 5.60-8.14 versus normal .36,
while old half-normal age stays 1.03-1.80 and fixed Student-t 6.05-6.14.
Learning residual level/variance helps beyond heavier tails or vanishing signal.

## Remaining Issue and Next Step

The normal validation resets above are spurious; fixed validation has no
physical failures. Relative-adaptive's unchanged heavy mode also resets in
20/20 late_second64 windows without endings there. Recovery is not universal.

Hold RL integration. Next run a compact causal spike-localization probe: save
peak transition index, current episode age, calibration group/threshold changes,
residual and recent predictive mean/variance on normal and persistent heavy
traces. Determine whether coarse age calibration boundaries or within-regime
variation cause reset, then fix using normal-only evidence before multi-switch
and matched-start/matched-budget RL controls. Do not expand training seeds yet.

## Limitations

Reused development replay, not improved return or publication readiness. The
adaptive model changes both location/variance, has 19-sample memory and uses a
real-valued approximation to positive scores. Later windows can include failures;
reset > .5 is descriptive, not a detection guarantee. Full arm/phase statistics:
`results/general_ood_v44_frozen_belief/summary.json`.

Follow-up: v45 localized calibration-boundary artifacts and separate normal
spikes; see `docs/general_ood_v45_reset_localization_results.md`. RL remains held.
