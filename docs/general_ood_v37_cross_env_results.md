# v37 complete results (2026-10-05)

All 24 cells have exit-0 completion and valid compact evaluation JSON. The
eight Hopper recovery runs completed on node007 and jtl110gpu2 without changing
commands or seeds. Pendulum task records are archived, not missing. Startup
failures remain in the manifest and are not counted as experimental samples.
The 24 evaluation files total 108,643 bytes; no checkpoints were downloaded.

## Main Result

Route B fails its frozen admission rule in both environments. The candidate
does not reproduce bus-like superiority. Returns below are averages of three
independent training-seed means; higher is better.

| Environment | Method | ID return | Markov 50x return |
|---|---|---:|---:|
| Pendulum | stable-context candidate | -909.15 | -1121.06 |
| Pendulum | no-context control | -616.01 | -965.57 |
| Pendulum | BAPR stable-context | -913.81 | -1097.42 |
| Pendulum | RE-SAC | -478.35 | -956.46 |
| Hopper | stable-context candidate | 865.07 | 468.48 |
| Hopper | no-context control | 994.54 | 651.76 |
| Hopper | BAPR stable-context | 877.47 | 664.17 |
| Hopper | RE-SAC | 1005.78 | 755.89 |

| OOD candidate minus comparator | Seed 0 | Seed 1 | Seed 2 | Mean | Paired 95% t interval | Wins |
|---|---:|---:|---:|---:|---|---|
| Pendulum minus BAPR | 170.39 | -76.25 | -165.04 | -23.63 | [-455.36, 408.10] | 1/3 |
| Pendulum minus RE-SAC | -80.02 | -270.07 | -143.70 | -164.60 | [-404.89, 75.70] | 0/3 |
| Hopper minus BAPR | -133.24 | -288.35 | -165.49 | -195.69 | [-399.01, 7.62] | 0/3 |
| Hopper minus RE-SAC | 267.60 | -765.37 | -364.46 | -287.41 | [-1581.08, 1006.26] | 1/3 |

The candidate loses every ID seed to RE-SAC in both environments. It also
loses the OOD mean to the no-context recipe in both environments. All paired
intervals above include zero; the admission failure is not a claim of a
population-level statistically significant disadvantage.

## Failure Localization

Hopper candidate OOD returns are 821.41, 250.91 and 333.12, with mean episode
lengths 843.85, 241.35 and 362.45 steps. In fixed `mega_event`, candidate
returns are 189.82, 124.42 and 151.48, versus BAPR's 955.74, 984.17 and 689.28.
The associated lengths are 219.0, 95.8 and 180.6 versus 962.4, 1000.0 and
568.4. This localizes a survival failure, not just a weak Markov average.
Pendulum's extreme fixed profiles are poor for all methods: `mega_event`
means span -1685.32 to -1675.04 and `recovery_aftershock` -1847.70 to -1805.08.

Hopper training interactions from completion logs, in seed order:

| Method | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| stable-context candidate | 1689884 | 1732454 | 1821848 | 1748062 |
| no-context control | 1740686 | 1581114 | 1723518 | 1681773 |
| BAPR stable-context | 1838033 | 1458689 | 1760827 | 1685850 |
| RE-SAC | 1515242 | 1746203 | 1803090 | 1688178 |

The candidate received more interactions on average than each comparator;
lower aggregate exposure does not explain its lower return.

## Limitations

The v38 checkpoint test found that v37's plain snapshots omit the context
normalizer state. Its context policies were evaluated with fresh statistics,
not their training input scale. The recorded scores and failed admission rule
are unchanged, but do not establish failure of correctly restored policies.
All presets disable SINDy/Jacobian losses. The no-context control also changes
BC and penalty weights; the MLP BAPR comparison changes actor optimization and
regularization as well as architecture. Neither isolates a single mechanism.
Hopper uses equal episodes rather than equal interactions, and no separate
Hopper cost is recorded. Three training seeds give wide descriptive intervals;
evaluation episodes are not independent training replicates.

## Next Step

Preserve v37 as a complete negative transfer result; keep paper claims
environment-conditional. Address the previously identified belief-response
direction issue separately, then freeze a small fixed-interaction comparison
that changes only context or the actor module with common optimization settings.
Do not expand seeds or tune v37 after seeing these outcomes. No new jobs were
launched. Full seed values and fixed-profile intervals are in
`results/general_ood_v37_cross_env/summary.json`.
