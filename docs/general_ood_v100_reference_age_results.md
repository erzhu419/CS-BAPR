# v100: age-balanced references offer local gains, not a coverage solution

Formal task t140769 completed successfully on node007 CPU in 59s: 62 windows, 3548 transitions and 186 incumbent prediction-error matches. Six local tests pass. Each recipe adds 4400 inputs from the existing v97 trajectories, with identical original support, coefficients, scaling and local-regression settings. Only scalar results were synchronized.

| Matched windows | Count | OOD suffix | Normal prefix | Age-balanced |
|---|---:|---:|---:|---:|
| v95 frozen | 30 | 0.060434 | 0.061711 | 0.061144 |
| v95 updating, changed modes | 24 | 0.162557 | 0.171306 | 0.156286 |
| v95 updating, normal | 6 | 0.048220 | 0.051299 | 0.042497 |
| Original v95 failure windows | 9 | 0.262948 | 0.283192 | 0.251983 |
| Six previously unsupported failures | 6 | 0.295242 | 0.295051 | 0.294579 |

Values are equal-window means of per-transition coordinate-RMS prediction errors, not pooled trajectory RMS. Age-balanced gains in changed updating windows concentrate in seed135190 (0.121626 to 0.103070); seed135191 changes only 0.149897 to 0.149464 and seed135192 worsens slightly, 0.216147 to 0.216323. Across those 24 windows, 11 improve and 13 worsen. Frozen windows worsen in 25/30 cases.

In the six unsupported failures, new-reference neighbor share rises from 0.0041% to 4.69% with age balancing, but mean prediction error falls only 0.22%. The historical v62 slippery control worsens from 0.140169 to 0.148724; its normal control improves from 0.042833 to 0.040524. Coverage remains poor in most selected failures; the changed neighbor population alone has not solved their prediction errors.

Next: retain the incumbent suffix bank and evaluate only age-balanced as an alternative through matched source-excluded error fitting, normal calibration and complete belief replay. Prefix-only is not selected. Follow-up v101 is dispatched; its execution state is recorded in experiments/general_ood_v101_age_replay_20261010/dispatch_manifest.json. Actor integration and seed expansion remain on HOLD.

## Limitations

One source policy, simulator-provided known-mode labels and already inspected development trajectories. Normal-prefix-only rows are temporally denser than suffix-only rows. This pilot evaluates means/support, not covariance scores, calibrated false-positive rates, recursive belief accuracy or RL returns. The two historical v62 windows lack oracle direction metrics. No new environment steps, optimizer updates or paper-table changes.
