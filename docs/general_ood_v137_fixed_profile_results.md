# v137 results and next step

All six tasks t144367--t144372 completed exit0. Each fresh policy trained300k
steps /299745 updates; total1.8M steps /1798470 updates. Matched protocol,
configuration, physical profiles and final-only selection verified.120 CPU
evaluation episodes /55203 steps. Only663622 bytes of formal scalar JSON
pulled; snapshots remain on servers. Native durations98.7--118.6 minutes.

| Seed | Fixed profile | Deterministic length /5 | Sampled length /15 | Sampled return |
|---|---|---:|---:|---:|
| 3 | ID | 1000.0 | 1000.0 | 3518.34 |
| 4 | ID | 1000.0 | 1000.0 | 3288.79 |
| 3 | Easy | 323.6 | 327.67 | 1214.87 |
| 4 | Easy | 287.8 | 281.33 | 1014.89 |
| 3 | Mega | 77.6 | 77.20 | 207.32 |
| 4 | Mega | 73.4 | 73.20 | 197.43 |

ID survives40/40 episodes; all80 non-ID episodes physically terminate.
All easy runs violate the height bound; mega runs violate height, sometimes
also angle. Mega final TD MSE0.302/0.116 is small despite poor survival.
Known-profile training has not passed the300k stable-control gate, so the
failure cannot be attributed solely to online context identification.

Next v138, planned not dispatched: isolate the existing mega physical factors
with mass-only3.6363, gravity-only2.6591, and friction-only0.37606, leaving
the other factors at1. Fresh seeds3/4,300k each: six runs /1.8M steps. Retain
the frozen training/config/reward/reset/noise protocol; reuse v137 ID and
compound-mega results as matched controls. The compound mass*gravity ratio
is9.6695 with unchanged actuators, motivating component rather than further
context sweeps or immediate random-regime training.

## Limitations
Two training seeds, this training configuration and300k budget only; failure
does not prove physical impossibility. No unknown-regime adaptation or strong
baseline superiority established. Paper tables unchanged. Summary:
`results/general_ood_v137_fixed_profile/summary.json`.
